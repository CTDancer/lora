import os
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import TrainerCallback

class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )       

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control