import numpy as np
import torch
import torch.nn.functional as F

import man.algorithms.classification.utils as cls_utils
import man.algorithms.fewshot.utils as fewshot_utils
import man.utils as utils
import pdb


def cluster_task(cluster_classifier, features, labels):
    """Applies the rotation prediction head to the given features."""
    scores = cluster_classifier(features)
    #pdb.set_trace()
    #labels = torch.Tensor(labels).cuda()
    assert scores.size(1) == 100
    loss = F.cross_entropy(scores, labels)
    

    return scores, loss

def object_classification_with_cluster_selfsupervision(
    feature_extractor,
    feature_extractor_optimizer,
    classifier,
    classifier_optimizer,
    classifier_clu,
    classifier_clu_optimizer,
    images,
    labels,
    is_train,
    alpha=1.0,
    base_ids=None,
    feature_name=None,
    images_unlabeled=None,
    images_unlabeled_label=None,
):
    """Forward-backward propagation routine for the classification model with
    the auxiliary rotation prediction task.

    Given as input a mini-batch of images with their labels, it applies the
    forward and (optionaly) backward propagation routines of a classification
    model extended to perfrom the auxiliary self-supervised rotation prediction
    task. The rotatation prediction task can optionally be applied to an extra
    mini-batch of only unlabeled images. The code assumes that the model is
    divided into a feature extractor, a classification head, and a rotation
    prediction head.

    Args:
    feature_extractor: The feature extractor neural network.
    feature_extractor_optimizer: The parameter optimizer of the feature
        extractor.
    classifier: The classification head applied on the output of the feature
        extractor.
    classifier_optimizer: The parameter optimizer of the classification head.
    images: A 4D tensor of shape [batch_size x channels x height x width] with
        the mini-batch images. It is assumed that this tensor is already on the
        same device as the feature extractor and classification head networks.
    labels: A 1D tensor with shape [batch_size] with the image labels. It is
        assumed that this tensor is already on the same device as the feature
        extractor and classification head networks.
    is_train: Boolean value that indicates if this mini-batch of images will be
        used for training or testing. If is_train is False, then the code does
        not apply the backward propagation step and does not update the
        parameter optimizers.
    alpha: The weight coeficient of the rotation prediction loss.
    base_ids: Optional argument used in case of episodic training of few-shot
        classification models. In this case, it is assumed that the total input
        batch_size consists of meta_batch_size training episodes, each with
        (batch_size // meta_batch_size) inner batch size (i.e., it must hold
        that batch_size % meta_batch_size == 0). In this context, base_ids is a
        2D tensor with shape [meta_batch_size x num_base], where base_ids[m] are
        the indices of the base categories that are being used in the m-th
        training episode.
    feature_name: (optional) A string or list of strings with the name of
        feature level(s) from which the feature extractor will extract features
        for the classification task.
    images_unlabeled: (optional) A 4D tensor of shape
        [batch_size2 x channels x height x width] with a mini-batch of unlabeled
        images. Only the rotation prediction task will be applied on those
        images.

    Returns:
    record: A dictionary of scalar values with the following items:
        'loss_cls': The cross entropy loss of the classification task.
        'loss_rot': The rotation prediction loss.
        'loss_total': The total loss, i.e., loss_cls + alpha * loss_rot.
        'Accuracy': The top-1 classification accuracy.
        'AccuracyRot': The rotation prediction accuracy.
    """

    if base_ids is not None:
        assert base_ids.size(0) == 1

    assert images.dim() == 4
    assert labels.dim() == 1
    assert images.size(0) == labels.size(0)

    if is_train:
        # Zero gradients.
        feature_extractor_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        classifier_clu_optimizer.zero_grad()

    batch_size_in = images.size(0)

    batch_size_classification = labels.size(0)
    record = {}
    with torch.set_grad_enabled(is_train):
        # Extract features from the images.
        features = cls_utils.extract_features(
            feature_extractor, images, feature_name=feature_name
        )

        # Perform the object classification task. From all the images only the
        # top 'batch_size_classification' are used for this task.
        features_cls = features[:batch_size_classification].contiguous()
        scores_classification, loss_classsification = cls_utils.classification_task(
            classifier, features_cls, labels, base_ids
        )
        record["loss_cls"] = loss_classsification.item()

        # Perform the self-supervised cluster prediction task.
        features_unlabeled = cls_utils.extract_features(
            feature_extractor, images_unlabeled, feature_name=feature_name
        )
        scores_cluster, loss_cluster = cluster_task(
            classifier_clu, features_unlabeled, images_unlabeled_label
        )
        record["loss_clu"] = loss_cluster.item()

        # Compute total loss.
        loss_total = loss_classsification + alpha * loss_cluster
        record["loss_total"] = loss_total.item()

    with torch.no_grad():
        # Compute accuracies.
        record["Accuracy"] = utils.top1accuracy(scores_classification, labels)
        record["AccuracyRot"] = utils.top1accuracy(scores_cluster, images_unlabeled_label)

    if is_train:
        # Backward loss and apply gradient steps.
        loss_total.backward()
        feature_extractor_optimizer.step()
        classifier_optimizer.step()
        classifier_clu_optimizer.step()

    return record


def fewshot_classification_with_rotation_selfsupervision(
    feature_extractor,
    feature_extractor_optimizer,
    classifier,
    classifier_optimizer,
    classifier_rot,
    classifier_rot_optimizer,
    images_train,
    labels_train,
    labels_train_1hot,
    images_test,
    labels_test,
    is_train,
    alpha=1.0,
    base_ids=None,
    feature_name=None,
):
    """Forward-backward routine of a few-shot model with auxiliary rotation
    prediction task.

    Given as input a mini-batch of few-shot episodes, it applies the
    forward and (optionally) backward propagation routines of the few-shot
    classification task. Each episode consists of (1) num_train_examples number
    of training examples for the novel classes of the few-shot episode, (2) the
    labels of training examples of the novel classes, (3) num_test_examples
    number of test examples of the few-shot episode (note that the test
    examples can be from both base and novel classes), and (4) the labels of the
    test examples. Each mini-batch consists of meta_batch_size number of
    few-shot episodes. The code assumes that the few-shot classification model
    is divided into a feature extractor network and a classification head
    network. Also, the code applies the auxiliary self-supervised task of
    predicting image rotations. The rotation prediction task is applied on both
    the test and train examples of the few-shot episodes.

    Args:
    feature_extractor: The feature extractor neural network.
    feature_extractor_optimizer: The parameter optimizer of the feature
        extractor. If None, then the feature extractor remains frozen during
        training.
    classifier: The classification head applied on the output of the feature
        extractor.
    classifier_optimizer: The parameter optimizer of the classification head.
    classifier_rot: The rotation prediction head applied on the output of the
        feature extractor.
    classifier_rot_optimizer: The parameter optimizer of the rotation prediction
        head.
    images_train: A 5D tensor with shape
        [meta_batch_size x num_train_examples x channels x height x width] that
        represents a mini-batch of meta_batch_size number of few-shot episodes,
        each with num_train_examples number of training examples.
    labels_train: A 2D tensor with shape
        [meta_batch_size x num_train_examples] that represents the discrete
        labels of the training examples of each few-shot episode in the batch.
    labels_train_1hot: A 3D tensor with shape
        [meta_batch_size x num_train_examples x num_novel] that represents
        the 1hot labels of the training examples of the novel classes of each
        few-shot episode in the batch. num_novel is the number of novel classes
        per few-shot episode.
    images_test: A 5D tensor with shape
        [meta_batch_size x num_test_examples x channels x height x width] that
        represents a mini-batch of meta_batch_size number of few-shot episodes,
        each with num_test_examples number of test examples.
    labels_test: A 2D tensor with shape
        [meta_batch_size x num_test_examples] that represents the discrete
        labels of the test examples of each few-shot episode in the mini-batch.
    is_train: Boolean value that indicates if this mini-batch will be
        used for training or testing. If is_train is False, then the code does
        not apply the backward propagation step and does not update the
        parameter optimizers.
    base_ids: A 2D tensor with shape [meta_batch_size x num_base], where
        base_ids[m] are the indices of the base categories that are being used
        in the m-th few-shot episode. num_base is the number of base classes per
        few-shot episode.
    alpha: (optional) The loss weight of the rotation prediction task.
    feature_name: (optional) A string or list of strings with the name of
        feature level(s) from which the feature extractor will extract features
        for the classification task.

    Returns:
    record: A dictionary of scalar values with the following items:
        'loss_cls': The cross entropy loss of the few-shot classification task.
        'loss_rot': The rotation prediction loss.
        'loss_total': The total loss, i.e., loss_cls + alpha * loss_rot.
        'AccuracyNovel': The classification accuracy of the test examples among
            only the novel classes.
        'AccuracyBase': (optinional) The classification accuracy of the test
            examples among only the base classes. Applicable, only if there are
            test examples from base classes in the mini-batch.
        'AccuracyBase': (optinional) The classification accuracy of the test
            examples among both the base and novel classes. Applicable, only if
            there are test examples from base classes in the mini-batch.
        'AccuracyRot': The accuracy of the rotation prediction task.
    """
    pdb.set_trace()
    assert images_train.dim() == 5
    assert images_test.dim() == 5
    assert images_train.size(0) == images_test.size(0)
    assert images_train.size(2) == images_test.size(2)
    assert images_train.size(3) == images_test.size(3)
    assert images_train.size(4) == images_test.size(4)
    assert labels_train.dim() == 2
    assert labels_test.dim() == 2
    assert labels_train.size(0) == labels_test.size(0)
    assert labels_train.size(0) == images_train.size(0)

    meta_batch_size = images_train.size(0)
    num_train = images_train.size(1)
    num_test = images_test.size(1)

    if is_train:  # zero the gradients
        feature_extractor_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        classifier_rot_optimizer.zero_grad()

    record = {}
    with torch.no_grad():
        images_train = utils.convert_from_5d_to_4d(images_train)
        images_test = utils.convert_from_5d_to_4d(images_test)
        labels_test = labels_test.view(-1)
        images = torch.cat([images_train, images_test], dim=0)

        batch_size_train = images_train.size(0)
        batch_size_train_test = images.size(0)
        assert batch_size_train == meta_batch_size * num_train
        assert batch_size_train_test == meta_batch_size * (num_train + num_test)

        # Create the 4 rotated version of the images; this step increases
        # the batch size by a multiple of 4.
        images = create_4rotations_images(images)
        labels_rotation = create_rotations_labels(batch_size_train_test, images.device)

    with torch.set_grad_enabled(is_train):
        # Extract features from the train and test images.
        features = cls_utils.extract_features(
            feature_extractor, images, feature_name=feature_name
        )

        # Apply the few-shot classification head.
        features_train = features[:batch_size_train]
        features_test = features[batch_size_train:batch_size_train_test]
        features_train = utils.add_dimension(features_train, meta_batch_size)
        features_test = utils.add_dimension(features_test, meta_batch_size)
        (
            classification_scores,
            loss_classsification,
        ) = fewshot_utils.few_shot_feature_classification(
            classifier,
            features_test,
            features_train,
            labels_train_1hot,
            labels_test,
            base_ids,
        )
        record["loss_cls"] = loss_classsification.item()

        # Apply the rotation prediction head.
        scores_rotation, loss_rotation = rotation_task(
            classifier_rot, features, labels_rotation
        )
        record["loss_rot"] = loss_rotation.item()

        # Compute total loss.
        loss_total = loss_classsification + alpha * loss_rotation
        record["loss_total"] = loss_total.item()

    with torch.no_grad():
        num_base = base_ids.size(1) if (base_ids is not None) else 0
        record = fewshot_utils.compute_accuracy_metrics(
            classification_scores, labels_test, num_base, record
        )
        record["AccuracyRot"] = utils.top1accuracy(scores_rotation, labels_rotation)

    if is_train:
        loss_total.backward()
        feature_extractor_optimizer.step()
        classifier_optimizer.step()
        classifier_rot_optimizer.step()

    return record


def object_classification_rotation_invariant(
    feature_extractor,
    feature_extractor_optimizer,
    classifier,
    classifier_optimizer,
    images,
    labels,
    is_train,
    rotation_invariant_classifier=False,
    random_rotation=False,
    base_ids=None,
    feature_name=None,
):
    """Applies the classification task to images augmented with rotations."""

    if base_ids is not None:
        assert base_ids.size(0) == 1

    if rotation_invariant_classifier:
        with torch.no_grad():
            if random_rotation:
                # randomly rotate an image.
                images, _ = randomly_rotate_images(images)
            else:
                # Create the 4 rotated version of each image using the same
                # label for each of them. The rotations are 0, 90, 180, and 270.
                images = create_4rotations_images(images)
                labels = labels.repeat(4)

    record = cls_utils.object_classification(
        feature_extractor,
        feature_extractor_optimizer,
        classifier,
        classifier_optimizer,
        images,
        labels,
        is_train,
        base_ids,
        feature_name,
    )

    return record
