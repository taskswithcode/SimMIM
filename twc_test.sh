inp_file=${1?"Specify input file"}
model_type=${2-1}

function run_model()
{
    inp_file=$1
    type=$2

    if [ $type -eq 1 ] 
    then
        echo "Running model Swin large"
        python -m torch.distributed.launch --nproc_per_node 1 main_finetune.py \
--eval --test_single $inp_file  --batch-size 1 --cfg configs/swin_large__800ep/simmim_finetune__swin_large__img224_window14__800ep.yaml --resume twc_models/simmim_finetune__swin_large__img224_window14__800ep.pth --data-path /home/acc/final_save/datasets/imagenet1k/ILSVRC/Data/CLS-LOC/ft
    else
        echo "Running model VIT base"
        python -m torch.distributed.launch --nproc_per_node 1 main_finetune.py \
--eval --test_single $inp_file --batch-size 1 --cfg configs/vit_base__800ep/simmim_finetune__vit_base__img224__800ep.yaml --resume twc_models/simmim_finetune__vit_base__img224__800ep.pth --data-path /home/acc/final_save/datasets/imagenet1k/ILSVRC/Data/CLS-LOC/ft
    fi

}

run_model $inp_file $model_type
