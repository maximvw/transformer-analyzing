.PHONY: data train-icot train-sft infer-icot infer-sft \
       fig2 fig3 fig5 fig6 fig7 table1 all-figures

# === Генерация данных ===
TRAIN_SIZE ?= 80800

data:
	uv run generate_data.py --train_size $(TRAIN_SIZE)

# === Тренировка моделей ===
N_LAYER ?= 2
N_HEAD ?= 4
N_EMBD ?= 768

TRAIN_CMD = TOKENIZERS_PARALLELISM=false uv run Internalize_CoT_Step_by_Step/src/train.py
TRAIN_BASE = --model gpt2 --n_layer $(N_LAYER) --n_head $(N_HEAD) --n_embd $(N_EMBD) \
	--train_path Internalize_CoT_Step_by_Step/data/4_by_4_mult/train.txt \
	--val_path Internalize_CoT_Step_by_Step/data/4_by_4_mult/valid.txt \
	--lr 5e-5 --batch_size 32 --seed 3456 --reset_optimizer

EPOCHS ?= 200
EPOCHS_SFT ?= 60

train-icot:
	$(TRAIN_CMD) $(TRAIN_BASE) \
		--epochs $(EPOCHS) \
		--remove_per_epoch 8 \
		--remove_all_when_remove_beyond inf \
		--removal_smoothing_lambda 4 \
		--removal_side left \
		--pretrain_epochs 0 \
		--save_model train_models/4_by_4_mult/icot

train-sft:
	$(TRAIN_CMD) $(TRAIN_BASE) \
		--epochs $(EPOCHS_SFT) \
		--remove_per_epoch 99999 \
		--removal_side left \
		--save_model train_models/4_by_4_mult/sft

# === Инференс ===
INFER_CMD = TOKENIZERS_PARALLELISM=false uv run Internalize_CoT_Step_by_Step/src/generate.py
TEST_PATH = Internalize_CoT_Step_by_Step/data/4_by_4_mult/test_bigbench.txt

infer-icot:
	$(INFER_CMD) \
		--from_pretrained train_models/4_by_4_mult/icot/checkpoint_12 \
		--test_path $(TEST_PATH) \
		--max_new_tokens 800

infer-sft:
	$(INFER_CMD) \
		--from_pretrained train_models/4_by_4_mult/sft/checkpoint_59 \
		--test_path $(TEST_PATH) \
		--max_new_tokens 800

# === Эксперименты (графики из статьи) ===
fig2:
	uv run icot/experiments/long_range_logit_attrib.py

fig3:
	uv run icot/experiments/probe_c_hat.py

fig5:
	uv run icot/experiments/fractals_and_minkowski.py

fig6:
	uv run icot/experiments/fourier_figure.py

fig7:
	uv run icot/experiments/grad_norms_and_losses.py

table1:
	uv run icot/experiments/fourier_r2_fits.py

all-figures: fig2 fig3 fig5 fig6 fig7 table1
