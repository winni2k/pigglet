from pigglet.tree_likelihood_mover import (
    TreeLikelihoodMover,
    PhyloTreeLikelihoodMover,
)

import numpy as np
import ray
import logging

logger = logging.getLogger(__name__)

if not ray.is_initialized():
    ray.init()


@ray.remote
class PhyloTreeLikelihoodMoverActor(PhyloTreeLikelihoodMover):
    def set_seed(self, seed):
        self.prng.seed(seed)

    def getattr(self, attr):
        return getattr(self, attr)

    def setattr(self, attr, *vals):
        setattr(self, attr, *vals)


def stride_ranges(num_items, num_chunks):
    cuts = [i / num_chunks for i in range(num_chunks + 1)]
    cuts = np.around(np.quantile(np.arange(num_items), cuts)).astype(np.int64)
    return cuts[0:-1], cuts[1:]


class PhyloTreeLikelihoodMoverDirector(TreeLikelihoodMover):
    def __init__(self, g, gls, prng, num_actors=2, testing: bool = False):
        self.calc = None
        self.mover = None
        self.testing = testing
        seed = prng.random()
        n_sites = gls.shape[0]
        self.actors = []
        if n_sites < num_actors:
            num_actors = n_sites
        g_id = ray.put(g)
        lefts, rights = stride_ranges(n_sites, num_actors)
        logger.info(f"Creating {len(lefts)} actors")
        for idx, (left, right) in enumerate(zip(lefts, rights)):
            logger.info(f"Actor {idx}: {right-left} sites")
            self.actors.append(
                PhyloTreeLikelihoodMoverActor.remote(  # type: ignore
                    g_id, gls[left:right], prng
                )
            )
            self.actors[-1].set_seed.remote(seed)

    @property
    def attachment_log_like(self):
        raise NotImplementedError
        # attachment_log_like = [
        #     ray.get(a.get_attachment_log_like.remote()) for a in self.actors
        # ]
        # logger.error(attachment_log_like)
        # return np.hstack(attachment_log_like)

    def random_move_and_get_like(self):
        likes = []
        for a in self.actors:
            likes.append(a.random_move_and_get_like.remote())
        return sum(ray.get(v) for v in likes)

    def make_and_register_random_move(self):
        for a in self.actors:
            a.make_and_register_random_move.remote()

    def has_changed_nodes(self):
        return self._get_attr_and_check_actors("has_changed_nodes")

    def undo(self):
        for actor in self.actors:
            actor.undo.remote()

    @property
    def move_changed_tree(self):
        return self._get_attr_and_check_actors("move_changed_tree")

    @property
    def double_check_ll_calculations(self):
        return self._get_attr_and_check_actors("double_check_ll_calculations")

    @double_check_ll_calculations.setter
    def double_check_ll_calculations(self, value):
        self._set_for_all_actors("double_check_ll_calculations", value)

    @property
    def check_logsumexp_accuracy(self):
        return self._get_attr_and_check_actors("check_logsumexp_accuracy")

    @check_logsumexp_accuracy.setter
    def check_logsumexp_accuracy(self, value):
        self._set_for_all_actors("check_logsumexp_accuracy", value)

    @property
    def logsumexp_refresh_rate(self):
        return self._get_attr_and_check_actors("logsumexp_refresh_rate")

    @logsumexp_refresh_rate.setter
    def logsumexp_refresh_rate(self, value):
        self._set_for_all_actors("logsumexp_refresh_rate", value)

    @property
    def g(self):
        return ray.get(self.actors[0].getattr.remote("g"))

    @property
    def mh_correction(self):
        return self._get_attr_and_check_actors("mh_correction")

    def log_likelihood(self):
        likelihoods = [a.log_likelihood.remote() for a in self.actors]
        return sum(ray.get(v) for v in likelihoods)

    @property
    def move_weights(self):
        return self._get_attr_and_check_actors("move_weights")

    def set_move_weight(self, weight_index, weight):
        for a in self.actors:
            a.set_move_weight.remote(weight_index, weight)

    def _get_attr_and_check_actors(self, actor_func_name):
        futures = [self.actors[0].getattr.remote(actor_func_name)]
        if self.testing:
            futures += [
                a.getattr.remote(actor_func_name) for a in self.actors[1:]
            ]
        first = ray.get(futures[0])
        if self.testing:
            assert all(ray.get(v) == first for v in futures[1:])
        return first

    def _set_for_all_actors(self, actor_func_name, *vals):
        for actor in self.actors:
            actor.setattr.remote(actor_func_name, *vals)

    def get_tracker_n_tries(self):
        return ray.get(self.actors[0].get_tracker_n_tries.remote())

    def get_tracker_acceptance_ratios(self):
        return ray.get(self.actors[0].get_tracker_acceptance_ratios.remote())

    def get_tracker_successful_proposal_time_proportions(self):
        return ray.get(
            self.actors[
                0
            ].get_tracker_successful_proposal_time_proportions.remote()
        )

    def flush_tracker(self):
        for actor in self.actors:
            actor.flush_tracker.remote()

    def get_calc_n_node_update_list(self):
        return ray.get(self.actors[0].get_calc_n_node_update_list.remote())

    def clear_calc_n_node_update_list(self):
        for actor in self.actors:
            actor.clear_calc_n_node_update_list.remote()

    def get_available_move_names(self):
        return ray.get(self.actors[0].get_available_move_names.remote())
