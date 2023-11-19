import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# if torch.has_mps:
#     print('Device: mps')
#     device = torch.device('mps')
# else:
#     print('Device: cpu')
#     device = torch.device('cpu')

# device = torch.device('cpu')


def visualize_heatmap(heatmap):
    with torch.no_grad():
        # Convert the input heatmap to a PyTorch tensor
        heatmap = torch.tensor(heatmap)

        # Move the heatmap to the device
        heatmap = heatmap.to(device)

        # Move the heatmap to the CPU
    heatmap = heatmap.cpu().numpy()

    # Stack the RGB channels
    true_heatmap = np.clip(heatmap.transpose(1, 2, 0), 0, 1)
    return true_heatmap
    # show
    plt.imshow(true_heatmap)
    plt.show()
    # Plot the heatmaps
    # fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))

    # ax1.imshow(true_heatmap)
    # ax1.set_title("True Heatmap")
    # ax1.axis("off")
    # plt.show()

def extract_peak(heatmap, max_pool_ks=7, min_score=0, max_det=30):
    """
    Extract local maxima (peaks) in a 2d heatmap.
    @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
    @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
    @min_score: Only return peaks greater than min_score
    @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
             heatmap value at the peak. Return no more than max_det peaks per image
    """

    if isinstance(heatmap, np.ndarray):
        heatmap_tensor = torch.tensor(heatmap)
    else:
        heatmap_tensor = heatmap

    heatmap_tensor = heatmap_tensor.to(device)

    if len(heatmap_tensor.shape) == 2:
        pooled_heatmap = F.max_pool2d(heatmap_tensor.unsqueeze(0).unsqueeze(0), max_pool_ks, stride=1, padding=max_pool_ks//2).squeeze()
    else:
        pooled_heatmap = F.max_pool2d(heatmap_tensor, max_pool_ks, stride=1, padding=max_pool_ks//2)

    heatmap_tensor = heatmap_tensor.cpu()
    pooled_heatmap = pooled_heatmap.cpu()
    peak_mask = torch.eq(heatmap_tensor, pooled_heatmap)
    peak_mask = torch.logical_and(peak_mask, heatmap_tensor > min_score)
    peak_indices = torch.nonzero(peak_mask, as_tuple=False)
    peak_scores = heatmap_tensor[peak_mask]
    peak_scores = peak_scores.detach().cpu()
    sorted_indices = torch.argsort(peak_scores, descending=True)
    sorted_indices = list(set(sorted_indices))
    sorted_indices = sorted_indices[:max_det]
    peaks = [(peak_scores[i].item(), peak_indices[i][1].item(), peak_indices[i][0].item()) for i in sorted_indices]
    return peaks


class Detector(torch.nn.Module):
    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))
        
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, bias=False)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=1)
            self.upconv = torch.nn.ConvTranspose2d(n_output, n_output, kernel_size=2, stride=1)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b3(self.c3(self.upconv(self.maxpool(self.b2(self.c2(self.upconv(self.maxpool(self.b1(self.c1(x)))))))))) + self.skip(x))


    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=3, kernel_size=3, use_skip=True):
        super().__init__()
        self.input_mean = torch.Tensor([0.2788, 0.2657, 0.2629])
        self.input_std = torch.Tensor([0.2064, 0.1944, 0.2252])

        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, self.Block(c, l, kernel_size, 2))
            c = l
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)

    def forward(self, x):
        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d'%i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d'%i](z)
            # Fix the padding
            # TODO: trying getting rid of this
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)
        return self.classifier(z)

    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        # Move the image to the device
        # image = image.to(device)

        # Initialize empty lists for the detections of each class
        detections_class_0 = []
        detections_class_1 = []
        detections_class_2 = []

        output_heatmaps = self.forward(image)[0]
        # save copy of output heatmaps
        # output_heatmaps = torch.sigmoid(output_heatmaps)
        # invert heatmaps
        # output_heatmaps = 1 - output_heatmaps

        # Loop through each class (channel) in the output heatmaps
        # heatmap = visualize_heatmap(output_heatmaps)
        # plt.imshow(heatmap)
        # plt.show()
        # # show image
        # plt.imshow(image.permute(1, 2, 0))
        # plt.show()
        for i in range(output_heatmaps.size(0)):
            # print(output_heatmaps[i].shape)
            # send it to csv
            # np.savetxt("heatmap.csv", output_heatmaps[i].detach().numpy(), delimiter=",")
            # Extract peaks from the heatmap for the current class
            j = (i + 1 ) % 3
            k = (i + 2) % 3
            peaks = extract_peak(output_heatmaps[i], min_score = 0, max_det = 30)
            if i == 0:
                detections_class_0.extend([(score, cx, cy, 0, 0) for (score, cx, cy) in peaks])
            elif i == 1:
                detections_class_1.extend([(score, cx, cy, 0, 0) for (score, cx, cy) in peaks])
            elif i == 2:
                detections_class_2.extend([(score, cx, cy, 0, 0) for (score, cx, cy) in peaks])

        # visualize_detection(image, [detections_class_0, detections_class_1, detections_class_2])
        # print('red detections')
        # print(detections_class_0)

        # print('green detections')
        # print(detections_class_1)

        # print('blue detections')
        # print(detections_class_2)
        return detections_class_0, detections_class_1, detections_class_2
        



import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_detection(image, detections):
    # Define colors for the circles
    colors = ['r', 'g', 'b']

    # Convert the image from torch tensor to numpy array
    image = image.permute(1, 2, 0).cpu().numpy()

    # Normalize the image back to the range [0, 1]
    image = image.astype(np.float32)

    # Create a figure and axis to display the image
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Loop through detections for each class
    for class_idx, class_detections in enumerate(detections):
        for score, cx, cy, w, h in class_detections:
            # Calculate the radius of the circle
            radius = max(w, h) / 2

            # Draw the circle
            circle = patches.Circle((cx, cy), radius, linewidth=2, edgecolor=colors[class_idx], facecolor='none')
            ax.add_patch(circle)

            # Add the class label and score to the circle
            label = f"Class {class_idx}: {score:.2f}"
            plt.text(cx - radius, cy - radius - 5, label, color=colors[class_idx], fontsize=8)

    # Display the image with detections
    plt.show()




def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    try:
        from .utils import DetectionSuperTuxDataset
    except:
        from utils import DetectionSuperTuxDataset

    dataset = DetectionSuperTuxDataset('../dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i+100]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        show()
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
    # dataset = DetectionSuperTuxDataset('../dense_data/valid', min_size=0)
    # import torchvision.transforms.functional as TF
    # from pylab import show, subplots
    # import matplotlib.patches as patches

    # fig, axs = subplots(1, 2)
    # model = load_model().eval().to(device)

    # im_filepath = '/Users/kylecox/Documents/UT MSCSO/CS 394D Deep Learning/homework4/dense_data/valid/00001_im.jpg'
    # # display image
    # axs[0].imshow(TF.to_pil_image(TF.to_tensor(plt.imread(im_filepath))), interpolation=None)
    # # plt.imshow(plt.imread(im_filepath))

    # # display detections
    # output = model.forward(TF.to_tensor(plt.imread(im_filepath)).unsqueeze(0).to(device))
    # output = torch.sigmoid(output)
    # # output = 1 - output
    # output = output.squeeze(0)
    # output = output.detach().cpu().numpy()

    # output = visualize_heatmap(output)
    # axs[1].imshow(output)

    # show()






    # aps = []
    # for i, ax in enumerate(axs.flat):
    #     im, kart, bomb, pickup = dataset[i]
    #     ax.imshow(TF.to_pil_image(im), interpolation=None)
    #     for k in kart:
    #         ax.add_patch(
    #             patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
    #     for k in bomb:
    #         ax.add_patch(
    #             patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
    #     for k in pickup:
    #         ax.add_patch(
    #             patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
    #     original_heatmap, *dets = model.detect(im)
    #     # visualize original heatmap
    #     # visualize_heatmap(original_heatmap)


    #     # fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    #     # for i in range(original_heatmap.size(0)):
    #     #     visualize_heatmap(original_heatmap[i].detach().cpu().numpy(), ax=axes[i])
    #     #     axes[i].set_title(f'Class {i} Heatmap')
    #     # im = TF.to_pil_image(im)
    #     # im = np.array(im)
    #     # visualize_heatmap(im, ax=axes[3])
    #     # axes[3].set_title('Original Image')
    #     # plt.show()

    #     # calculate average precision for each color value c
    #     ap_images = []
    #     for c in range(3):
    #         y_true = []
    #         y_score = []
    #         for s, cx, cy, w, h in dets[c]:
    #             if any((cx > k[0] and cx < k[2] and cy > k[1] and cy < k[3]) for k in kart):
    #                 y_true.append(1)
    #             else:
    #                 y_true.append(0)
    #             y_score.append(s)
    #         if len(y_true) == 0:
    #             ap = 0
    #         else:
    #             print(y_true, y_score)
    #             ap = average_precision_score(y_true, y_score)
    #         ap_images.append(ap)
    #         print(f'Average precision for color {c}: {ap}')

    #         for s, cx, cy, w, h in dets[c]:
    #             if any((cx > k[0] and cx < k[2] and cy > k[1] and cy < k[3]) for k in kart):
    #                 ax.add_patch(patches.Rectangle((cx - 0.5, cy - 0.5), 1, 1, facecolor='none', edgecolor='r'))
    #             else:
    #                 ax.add_patch(patches.Rectangle((cx - 0.5, cy - 0.5), 1, 1, facecolor='none', edgecolor='g'))
    #     aps.append(ap_images)
    # show()

                   


