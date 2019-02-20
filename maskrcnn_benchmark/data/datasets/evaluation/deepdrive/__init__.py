import logging

from .deepdrive_eval import do_deepdrive_evaluation


def deepdrive_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("deepdrive evaluation doesn't support box_only, ignored.")
    logger.info("performing deepdrive evaluation, ignored iou_types.")
    return do_deepdrive_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
