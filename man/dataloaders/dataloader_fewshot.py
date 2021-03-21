import math
import random

import numpy as np
import torch
import torchnet as tnt
import pdb

class FewShotDataloader:
    def __init__(
        self,
        dataset,
        nKnovel=5,
        nKbase=-1,
        n_exemplars=1,
        n_test_novel=15 * 5,
        n_test_base=15 * 5,
        batch_size=1,
        num_workers=4,
        epoch_size=2000,
    ):

        self.dataset = dataset
        self.phase = self.dataset.phase
        max_possible_nKnovel = (
            self.dataset.num_cats_base
            if (self.phase == "train" or self.phase == "trainval" or self.phase == "train_not_miniimagnet")
            else self.dataset.num_cats_novel
        )

        assert 0 <= nKnovel <= max_possible_nKnovel
        self.nKnovel = nKnovel

        max_possible_nKbase = self.dataset.num_cats_base
        nKbase = nKbase if nKbase >= 0 else max_possible_nKbase
        if (self.phase == "train" or self.phase == "trainval" or self.phase == "train_not_miniimagnet") and nKbase > 0:
            nKbase -= self.nKnovel
            max_possible_nKbase -= self.nKnovel
        #pdb.set_trace()

        assert 0 <= nKbase <= max_possible_nKbase
        self.nKbase = nKbase
        self.n_exemplars = n_exemplars
        self.n_test_novel = n_test_novel
        self.n_test_base = n_test_base
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase == "test") or (self.phase == "val")

    def sample_image_ids_from(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset.label2ind[cat_id]).

        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.

        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
        """
        #pdb.set_trace()
        assert cat_id in self.dataset.label2ind.keys()
        assert len(self.dataset.label2ind[cat_id]) >= sample_size
        # Note: random.sample samples elements without replacement.
        return random.sample(self.dataset.label2ind[cat_id], sample_size)

    def sample_categories(self, cat_set, sample_size=1):
        """
        Samples `sample_size` number of unique categories picked from the
        `cat_set` set of categories. `cat_set` can be either 'base' or 'novel'.

        Args:
            cat_set: string that specifies the set of categories from which
                categories will be sampled.
            sample_size: number of categories that will be sampled.

        Returns:
            cat_ids: a list of length `sample_size` with unique category ids.
        """
        if cat_set == "base":
            labelIds = self.dataset.labelIds_base
        elif cat_set == "novel":
            labelIds = self.dataset.labelIds_novel
        else:
            raise ValueError(f"Not recognized category set {cat_set}")

        assert len(labelIds) >= sample_size
        # return sample_size unique categories chosen from labelIds set of
        # categories (that can be either self.labelIds_base or self.labelIds_novel)
        # Note: random.sample samples elements without replacement.
        return random.sample(labelIds, sample_size)

    def sample_base_and_novel_categories(self, nKbase, nKnovel):
        """
        Samples `nKbase` number of base categories and `nKnovel` number of novel
        categories.

        Args:
            nKbase: number of base categories
            nKnovel: number of novel categories

        Returns:
            Kbase: a list of length 'nKbase' with the ids of the sampled base
                categories.
            Knovel: a list of lenght 'nKnovel' with the ids of the sampled novel
                categories.
        """
        if self.is_eval_mode:
            assert nKnovel <= self.dataset.num_cats_novel
            # sample from the set of base categories 'nKbase' number of base
            # categories.
            Kbase = sorted(self.sample_categories("base", nKbase))
            # sample from the set of novel categories 'nKnovel' number of novel
            # categories.
            Knovel = sorted(self.sample_categories("novel", nKnovel))
        else:
            # sample from the set of base categories 'nKnovel' + 'nKbase' number
            # of categories.
            cats_ids = self.sample_categories("base", nKnovel + nKbase)
            assert len(cats_ids) == (nKnovel + nKbase)
            # Randomly pick 'nKnovel' number of fake novel categories and keep
            # the rest as base categories.
            random.shuffle(cats_ids)
            Knovel = sorted(cats_ids[:nKnovel])
            Kbase = sorted(cats_ids[nKnovel:])

        return Kbase, Knovel

    def sample_test_examples_for_base_categories(self, Kbase, n_test_base):
        """
        Sample `n_test_base` number of images from the `Kbase` categories.

        Args:
            Kbase: a list of length `nKbase` with the ids of the categories from
                where the images will be sampled.
            n_test_base: the total number of images that will be sampled.

        Returns:
            Tbase: a list of length `n_test_base` with 2-element tuples. The 1st
                element of each tuple is the image id that was sampled and the
                2nd elemend is its category label (which is in the range
                [0, len(Kbase)-1]).
        """
        Tbase = []
        if len(Kbase) > 0:
            # Sample for each base category a number images such that the total
            # number sampled images of all categories to be equal to `n_test_base`.
            KbaseIndices = np.random.choice(
                np.arange(len(Kbase)), size=n_test_base, replace=True
            )
            KbaseIndices, NumImagesPerCategory = np.unique(KbaseIndices, return_counts=True)
            for Kbase_idx, NumImages in zip(KbaseIndices, NumImagesPerCategory):
                imd_ids = self.sample_image_ids_from(Kbase[Kbase_idx], sample_size=NumImages)
                Tbase += [(img_id, Kbase_idx) for img_id in imd_ids]

        assert len(Tbase) == n_test_base

        return Tbase

    def sample_train_and_test_examples_for_novel_categories(
        self, Knovel, nTestExamplesTotal, n_exemplars, nKbase
    ):
        """Samples train and test examples of the novel categories.

        Args:
    	    Knovel: a list with the ids of the novel categories.
            nTestExamplesTotal: the total number of test images that will be sampled
                from all the novel categories.
            n_exemplars: the number of training examples per novel category that
                will be sampled.
            nKbase: the number of base categories. It is used as offset of the
                category index of each sampled image.

        Returns:
            Tnovel: a list of length `n_test_novel` with 2-element tuples. The
                1st element of each tuple is the image id that was sampled and
                the 2nd element is its category label (which is in the range
                [nKbase, nKbase + len(Knovel) - 1]).
            Exemplars: a list of length len(Knovel) * n_exemplars of 2-element
                tuples. The 1st element of each tuple is the image id that was
                sampled and the 2nd element is its category label (which is in
                the ragne [nKbase, nKbase + len(Knovel) - 1]).
        """

        if len(Knovel) == 0:
            return [], []

        nKnovel = len(Knovel)
        Tnovel = []
        Exemplars = []

        assert (nTestExamplesTotal % nKnovel) == 0
        nTestExamples = nTestExamplesTotal // nKnovel

        for Knovel_idx in range(len(Knovel)):
            img_ids = self.sample_image_ids_from(
                Knovel[Knovel_idx], sample_size=(nTestExamples + n_exemplars)
            )

            img_labeled = img_ids[: (nTestExamples + n_exemplars)]
            img_tnovel = img_labeled[:nTestExamples]
            img_exemplars = img_labeled[nTestExamples:]

            Tnovel += [(img_id, nKbase + Knovel_idx) for img_id in img_tnovel]
            Exemplars += [(img_id, nKbase + Knovel_idx) for img_id in img_exemplars]

        assert len(Tnovel) == nTestExamplesTotal
        assert len(Exemplars) == len(Knovel) * n_exemplars
        random.shuffle(Exemplars)

        return Tnovel, Exemplars

    def sample_episode(self):
        """Samples a training episode."""
        nKnovel = self.nKnovel
        nKbase = self.nKbase
        n_test_novel = self.n_test_novel
        n_test_base = self.n_test_base
        n_exemplars = self.n_exemplars

        Kbase, Knovel = self.sample_base_and_novel_categories(nKbase, nKnovel)
        #pdb.set_trace()
        Tbase = self.sample_test_examples_for_base_categories(Kbase, n_test_base)
        outputs = self.sample_train_and_test_examples_for_novel_categories(
            Knovel, n_test_novel, n_exemplars, nKbase
        )
        Tnovel, Exemplars = outputs

        # concatenate the base and novel category examples.
        Test = Tbase + Tnovel
        random.shuffle(Test)
        Kall = Kbase + Knovel
        return Exemplars, Test, Kall, nKbase

    def create_examples_tensor_data(self, examples):
        """
        Creates the examples image and label tensor data.

        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).

        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """
        images = torch.stack([self.dataset[img_idx][0] for img_idx, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(_):
            Exemplars, Test, Kall, nKbase = self.sample_episode()
            Xt, Yt = self.create_examples_tensor_data(Test)
            Kall = torch.LongTensor(Kall)
            if len(Exemplars) > 0:
                Xe, Ye = self.create_examples_tensor_data(Exemplars)
                return Xe, Ye, Xt, Yt, Kall, nKbase
            else:
                return Xt, Yt, Kall, nKbase

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=list(range(self.epoch_size)), load=load_function
        )
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=(False if self.is_eval_mode else True),
        )

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size // self.batch_size


class FewShotDataloaderWithPatches(FewShotDataloader):
    def __init__(
        self,
        dataset,
        dataset_patches,
        nKnovel=5,
        nKbase=-1,
        n_exemplars=1,
        n_test_novel=15 * 5,
        n_test_base=15 * 5,
        batch_size=1,
        num_workers=4,
        epoch_size=2000,
    ):

        super().__init__(
            dataset=dataset,
            nKnovel=nKnovel,
            nKbase=nKbase,
            n_exemplars=n_exemplars,
            n_test_novel=n_test_novel,
            n_test_base=n_test_base,
            batch_size=batch_size,
            num_workers=num_workers,
            epoch_size=epoch_size,
        )

        self.dataset_patches = dataset_patches

    def create_examples_tensor_patch_data(self, examples):
        """
        Creates the examples image and label tensor data.

        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).

        Returns:
            images: a tensor of shape [nExamples, 9, Height, Width, 3] with the
                9 patches of each example image, where nExamples is the number
                of examples (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """
        images = torch.stack(
            [self.dataset_patches[img_idx][0] for img_idx, _ in examples], dim=0
        )
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(_):
            Exemplars, Test, Kall, nKbase = self.sample_episode()
            Xt, Yt = self.create_examples_tensor_data(Test)
            Xt_patches, _ = self.create_examples_tensor_patch_data(Test)
            Kall = torch.LongTensor(Kall)
            if len(Exemplars) > 0:
                Xe, Ye = self.create_examples_tensor_data(Exemplars)
                Xe_patches, _ = self.create_examples_tensor_patch_data(Exemplars)
                return Xe, Xe_patches, Ye, Xt, Xt_patches, Yt, Kall, nKbase
            else:
                return Xt, Xt_patches, Yt, Kall, nKbase

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=list(range(self.epoch_size)), load=load_function
        )
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=(False if self.is_eval_mode else True),
        )

        return data_loader


class LowShotDataloader:
    def __init__(
        self,
        dataset_train_novel,
        dataset_evaluation,
        n_exemplars=1,
        batch_size=1,
        num_workers=4,
    ):

        self.n_exemplars = n_exemplars
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_train_novel = dataset_train_novel
        self.dataset_evaluation = dataset_evaluation

        assert (
            self.dataset_evaluation.labelIds_novel == self.dataset_train_novel.labelIds_novel
        )

        assert self.dataset_evaluation.labelIds_base == self.dataset_train_novel.labelIds_base

        assert (
            self.dataset_evaluation.base_classes_eval_split
            == self.dataset_train_novel.base_classes_eval_split
        )

        self.nKnovel = self.dataset_evaluation.num_cats_novel
        self.nKbase = self.dataset_evaluation.num_cats_base

        # Category ids of the base categories.
        self.Kbase = sorted(self.dataset_evaluation.labelIds_base)
        assert self.nKbase == len(self.Kbase)
        # Category ids of the novel categories.
        self.Knovel = sorted(self.dataset_evaluation.labelIds_novel)

        assert self.nKnovel == len(self.Knovel)
        self.Kall = self.Kbase + self.Knovel

        self.CategoryId2LabelIndex = {
            category_id: label_index for label_index, category_id in enumerate(self.Kall)
        }
        self.Kbase_eval_split = self.dataset_train_novel.base_classes_eval_split

        Kbase_set = set(self.Kall[: self.nKbase])
        Kbase_eval_split_set = set(self.Kbase_eval_split)
        assert len(set.intersection(Kbase_set, Kbase_eval_split_set)) == len(
            Kbase_eval_split_set
        )

        self.base_eval_split_labels = sorted(
            [self.CategoryId2LabelIndex[category_id] for category_id in self.Kbase_eval_split]
        )

        # Collect the image indices of the evaluation set for both the base and
        # the novel categories.
        data_indices = []
        for category_id in self.Kbase_eval_split:
            data_indices += self.dataset_evaluation.label2ind[category_id]
        for category_id in self.Knovel:
            data_indices += self.dataset_evaluation.label2ind[category_id]
        self.eval_data_indices = sorted(data_indices)
        self.epoch_size = len(self.eval_data_indices)

    def base_category_label_indices(self):
        return self.base_eval_split_labels

    def novel_category_label_indices(self):
        return list(range(self.nKbase, len(self.Kall)))

    def sample_image_ids_from(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset_train_novel.label2ind[cat_id]).

        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.

        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
        """
        assert cat_id in self.dataset_train_novel.label2ind
        assert len(self.dataset_train_novel.label2ind[cat_id]) >= sample_size
        # Note: random.sample samples elements without replacement.
        return random.sample(self.dataset_train_novel.label2ind[cat_id], sample_size)

    def sample_training_examples_for_novel_categories(self, Knovel, n_exemplars, nKbase):
        """Samples (a few) training examples for the novel categories.

        Args:
            Knovel: a list with the ids of the novel categories.
            n_exemplars: the number of training examples per novel category.
            nKbase: the number of base categories.

        Returns:
            Exemplars: a list of length len(Knovel) * n_exemplars of 2-element
                tuples. The 1st element of each tuple is the image id that was
                sampled and the 2nd element is its category label (which is in
                the ragne [nKbase, nKbase + len(Knovel) - 1]).
        """
        Exemplars = []
        for knovel_idx, knovel_label in enumerate(Knovel):
            imds = self.sample_image_ids_from(knovel_label, sample_size=n_exemplars)
            Exemplars += [(img_id, nKbase + knovel_idx) for img_id in imds]
        random.shuffle(Exemplars)

        return Exemplars

    def create_examples_tensor_data(self, examples):
        """
        Creates the examples image and label tensor data.

        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).

        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """
        images = torch.stack(
            [self.dataset_train_novel[img_idx][0] for img_idx, _ in examples], dim=0
        )
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def sample_training_data_for_novel_categories(self, exp_id=0):
        nKnovel = self.nKnovel
        nKbase = self.nKbase
        random.seed(exp_id)  # fix the seed for this experiment.
        # Sample `n_exemplars` number of training examples per novel category.
        train_examples = self.sample_training_examples_for_novel_categories(
            self.Knovel, self.n_exemplars, nKbase
        )
        Kall = torch.LongTensor(self.Kall)
        images_train, labels_train = self.create_examples_tensor_data(train_examples)

        return images_train, labels_train, Kall, nKbase, nKnovel

    def get_iterator(self, epoch=0):
        def load_fun_(idx):
            img_idx = self.eval_data_indices[idx]
            img, category_id = self.dataset_evaluation[img_idx]
            label = (
                self.CategoryId2LabelIndex[category_id]
                if (category_id in self.CategoryId2LabelIndex)
                else -1
            )
            return img, label

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=list(range(self.epoch_size)), load=load_fun_
        )
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return int(math.ceil(float(self.epoch_size) / self.batch_size))
