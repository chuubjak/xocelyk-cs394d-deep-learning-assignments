import pystk
import numpy as np


def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    # if current_vel < 10 and abs(aim_point[0]) > 0.9:
    #     aim_point[0] = 0

    action.steer = aim_point[0] * 15/(current_vel + 1)
    action.steer = max(min(action.steer, 1), -1)


    target_vel = 35 / ((abs(action.steer) + 1)**2)
    # target_vel = 25 / ((abs(action.steer) + 1)**1.5)
    accel_factor = .5
    if current_vel < target_vel:
        action.acceleration = (target_vel - current_vel) * accel_factor
        if action.acceleration > 6:
            action.nitro = True
    elif current_vel > target_vel:
        action.acceleration = 0
        action.brake = True
    # print(round(current_vel, 2), round(target_vel, 2), round(action.steer, 2), round(action.acceleration, 2))
    drift_threshold = 0.23
    # drift_threshold = 0.18
    # drift_threshold = 0.15
    # drift_threshold = 0.3
    action.drift = abs(aim_point[0]) > drift_threshold
    # action.steer += np.random.normal(0, 0.01)

    return action


def rollout(self, track, controller, planner=None, max_frames=1000, verbose=False, data_callback=None):
        """
        Play a level (track) for a single round.
        :param track: Name of the track
        :param controller: low-level controller, see controller.py
        :param planner: high-level planner, see planner.py
        :param max_frames: Maximum number of frames to play for
        :param verbose: Should we use matplotlib to show the agent drive?
        :param data_callback: Rollout calls data_callback(time_step, image, 2d_aim_point) every step, used to store the
                              data
        :return: Number of steps played
        """

        import io

        if self.k is not None and self.k.config.track == track:
            self.k.restart()
            self.k.step()
        else:
            if self.k is not None:
                self.k.stop()
                del self.k
            config = pystk.RaceConfig(num_kart=1, laps=1, track=track)
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

            self.k = pystk.Race(config)
            self.k.start()
            self.k.step()

        state = pystk.WorldState()
        track = pystk.Track()

        last_rescue = 0

        if verbose:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1)

        frames = []

        for t in range(max_frames):

            state.update()
            track.update()

            kart = state.players[0].kart

            if np.isclose(kart.overall_distance / track.length, 1.0, atol=2e-3):
                if verbose:
                    print("Finished at t=%d" % t)
                break

            proj = np.array(state.players[0].camera.projection).T
            view = np.array(state.players[0].camera.view).T

            aim_point_world = self._point_on_track(kart.distance_down_track+TRACK_OFFSET, track)
            aim_point_image = self._to_image(aim_point_world, proj, view)
            if data_callback is not None:
                data_callback(t, np.array(self.k.render_data[0].image), aim_point_image)

            if planner:
                image = np.array(self.k.render_data[0].image)
                aim_point_image = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()

            current_vel = np.linalg.norm(kart.velocity)
            action = controller(aim_point_image, current_vel)

            if current_vel < 1.0 and t - last_rescue > RESCUE_TIMEOUT:
                last_rescue = t
                action.rescue = True

            if verbose:
                ax.clear()
                ax.imshow(self.k.render_data[0].image)
                WH2 = np.array([self.config.screen_width, self.config.screen_height]) / 2
                ax.add_artist(plt.Circle(WH2*(1+self._to_image(kart.location, proj, view)), 2, ec='b', fill=False, lw=1.5))
                ax.add_artist(plt.Circle(WH2*(1+self._to_image(aim_point_world, proj, view)), 2, ec='r', fill=False, lw=1.5))
                if planner:
                    ap = self._point_on_track(kart.distance_down_track + TRACK_OFFSET, track)
                    ax.add_artist(plt.Circle(WH2*(1+aim_point_image), 2, ec='g', fill=False, lw=1.5))
                plt.pause(1e-3)

                with io.BytesIO() as buff:
                    fig.savefig(buff, format='raw')
                    buff.seek(0)
                    data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                im = data.reshape((int(h), int(w), -1))

                frames.append(im)

            self.k.step(action)
            t += 1

        if verbose:
            import imageio

            imageio.mimwrite("test.mp4", frames, fps=30, bitrate=1000000)
            return t, kart.overall_distance / track.length

if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            how_far = np.round(how_far, 2)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
