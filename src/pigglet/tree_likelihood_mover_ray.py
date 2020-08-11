from pigglet.tree_likelihood_mover import (
    TreeLikelihoodMover,
    PhyloTreeLikelihoodMover,
)

import ray
import numpy as np

if not ray.is_initialized():
    ray.init()


@ray.remote
class PhyloTreeLikelihoodMoverActor(PhyloTreeLikelihoodMover):
    def set_seed(self, seed):
        self.prng.seed(seed)

    def get_attachment_log_like(self):
        return self.attachment_log_like

    def get_has_changed_nodes(self):
        return self.has_changed_nodes

    def get_double_check_ll_calculations(self):
        return self.double_check_ll_calculations

    def set_double_check_ll_calculations(self, val):
        self.double_check_ll_calculations = val

    def get_check_logsumexp_accuracy(self):
        return self.check_logsumexp_accuracy

    def set_check_logsumexp_accuracy(self, val):
        self.check_logsumexp_accuracy = val

    def get_logsumexp_refresh_rate(self):
        return self.logsumexp_refresh_rate

    def set_logsumexp_refresh_rate(self, val):
        self.logsumexp_refresh_rate = val

    def get_g(self):
        return self.g


def stride_ranges(num_items, num_chunks):
    min_stride = num_items // num_chunks
    left_over = num_items % num_chunks
    strides = [0]
    for i in range(num_chunks):
        assert left_over < num_chunks
        strides.append(strides[-1] + min_stride)
        if left_over != 0:
            strides[-1] = 1
            left_over -= 1
    strides_l = strides[:-1]
    strides_r = strides[1:]
    return strides_l, strides_r


class PhyloTreeLikelihoodMoverDirector(TreeLikelihoodMover):
    def __init__(self, g, gls, prng, num_actors=2):
        super().__init__()
        seed = prng.random()
        n_sites = gls.shape[0]
        self.actors = []
        if n_sites > num_actors:
            num_actors = n_sites
        g_id = ray.put(g)
        for left, right in zip(*stride_ranges(n_sites, num_actors)):
            self.actors.append(
                PhyloTreeLikelihoodMoverActor.remote(
                    g_id, gls[left:right], prng
                )
            )
            self.actors[-1].set_seed.remote(seed)

    @property
    def attachment_log_like(self):
        attachment_log_like = [
            ray.get(a.get_attachment_log_like.remote()) for a in self.actors
        ]
        return np.vstack(attachment_log_like)

    def random_move(self, weights=None):
        for a in self.actors:
            a.random_move.remote(weights)

    def has_changed_nodes(self):
        return self._get_and_check_actors("get_has_changed_nodes")

    def undo(self):
        for actor in self.actors:
            actor.undo.remote()

    def register_mh_result(self, accepted: bool):
        self._set_for_all_actors("register_mh_result", accepted)

    @property
    def double_check_ll_calculations(self):
        return self._get_and_check_actors("get_double_check_ll_calculations")

    @double_check_ll_calculations.setter
    def double_check_ll_calculations(self, value):
        self._set_for_all_actors("set_double_check_ll_calculations", value)

    @property
    def check_logsumexp_accuracy(self):
        return self._get_and_check_actors("get_check_logsumexp_accuracy")

    @check_logsumexp_accuracy.setter
    def check_logsumexp_accuracy(self, value):
        self._set_for_all_actors("set_check_logsumexp_accuracy", value)

    @property
    def logsumexp_refresh_rate(self):
        return self._get_and_check_actors("get_logsumexp_refresh_rate")

    @logsumexp_refresh_rate.setter
    def logsumexp_refresh_rate(self, value):
        for actor in self.actors:
            actor.set_logsumexp_refresh_rate.remote(value)

    @property
    def g(self):
        return ray.get(self.actors[0].get_g.remote())

    def log_likelihood(self):
        likelihoods = [a.log_likelihood.remote() for a in self.actors]
        return sum(ray.get(v) for v in likelihoods)

    def _get_and_check_actors(self, actor_func_name):
        futures = [getattr(a, actor_func_name).remote() for a in self.actors]
        first = ray.get(futures[0])
        assert all(ray.get(v) == first for v in futures)
        return first

    def _set_for_all_actors(self, actor_func_name, val):
        for actor in self.actors:
            getattr(actor, actor_func_name).remote(val)
