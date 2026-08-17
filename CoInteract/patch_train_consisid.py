import re, ast, sys
src = "examples/wanvideo/model_training/train.py"
dst = "train_consisid.py"
s = open(src).read()

# 1) use the COPIED pipeline (has the model_fn_wans2v injection)
s = s.replace(
    "from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig",
    "from diffsynth.pipelines.wan_video_new_consisid import WanVideoPipeline, ModelConfig", 1)

# 2) attach ConsisID modules (trainable) after the MoE-training block, gated by CONSISID=1
moe_anchor = '            print(f"[MoE] {moe_param_count:,} params set to full training")\n'
assert moe_anchor in s, "MoE anchor not found"
attach = moe_anchor + (
"\n        # --- ConsisID: attach identity pathway; train ONLY these (base frozen) ---\n"
"        if os.environ.get('CONSISID', '0') == '1':\n"
"            from diffsynth.models.consisid_faithful import LocalFacialExtractor, PerceiverCrossAttention\n"
"            import torch.nn as _cnn\n"
"            _d = self.pipe.dit\n"
"            _p0 = next(_d.parameters()); _dt = _p0.dtype; _dev = _p0.device\n"
"            _d.consisid_interval = 2\n"
"            _d.consisid_extractor = LocalFacialExtractor(id_dim=512, vit_dim=1280, depth=4, num_scale=1, num_queries=32, output_dim=2048).to(_dev, _dt)\n"
"            _nca = len(_d.blocks) // 2\n"
"            _d.consisid_cross_attn = _cnn.ModuleList([PerceiverCrossAttention(dim=5120, kv_dim=2048).to(_dev, _dt) for _ in range(_nca)])\n"
"            _d._consisid_enabled = True\n"
"            _cp = 0\n"
"            for _pp in _d.consisid_extractor.parameters(): _pp.requires_grad = True; _cp += _pp.numel()\n"
"            for _pp in _d.consisid_cross_attn.parameters(): _pp.requires_grad = True; _cp += _pp.numel()\n"
"            self._consisid = True\n"
"            print(f'[ConsisID] attached + trainable: {_cp/1e6:.0f}M params ({_nca} cross-attn blocks)')\n"
"        else:\n"
"            self._consisid = False\n"
)
s = s.replace(moe_anchor, attach, 1)

# 3) in forward(): compute id_tokens (CLIP-only v1) and set dit._consisid_id_tokens before training_loss
loss_anchor = "        loss = self.pipe.training_loss(\n"
assert loss_anchor in s, "training_loss anchor not found"
idset = (
"        if getattr(self, '_consisid', False):\n"
"            _d = self.pipe.dit\n"
"            _p0 = next(_d.parameters()); _dt = _p0.dtype; _dev = _p0.device\n"
"            _pi = data['person_image']\n"
"            _clip = self.pipe.image_encoder.encode_image([self.pipe.preprocess_image(_pi)]).to(_dev, _dt)\n"
"            import torch as _t\n"
"            _arc = _t.zeros(_clip.shape[0], 512, device=_dev, dtype=_dt)  # CLIP-only v1 (ArcFace zeroed)\n"
"            _d._consisid_id_tokens = _d.consisid_extractor(_arc, [_clip])\n"
) + loss_anchor
s = s.replace(loss_anchor, idset, 1)

ast.parse(s)
open(dst, "w").write(s)
print("train_consisid.py written + syntax OK")
