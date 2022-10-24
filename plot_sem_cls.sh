python tools/analysis_tools/analyze_logs.py \
plot_curve /home/gabriel/mmdetection3d/work_dirs/3detr-m/20221023_232148.log.json \
--keys loss_sem_cls loss_sem_cls_0 loss_sem_cls_1 loss_sem_cls_2 loss_sem_cls_3 loss_sem_cls_4 loss_sem_cls_5 loss_sem_cls_6 \
--legend loss_sem_cls loss_sem_cls_0 loss_sem_cls_1 loss_sem_cls_2 loss_sem_cls_3 loss_sem_cls_4 loss_sem_cls_5 loss_sem_cls_6 \
--out sem_cls.pdf