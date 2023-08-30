import numpy as np
import timm  # pip install timm
import torch
from vit import ViT

# Helpers
def get_n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2):
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()

    np.testing.assert_allclose(a1, a2)

model_name = "vit_base_patch16_384"
official_model = timm.create_model(model_name, pretrained=True)
official_model.eval()
print(type(official_model))

custom_config = {"img_size": 384,
                 "in_chans": 3,
                 "patch_size": 16,
                 "embed_dim": 768,
                 "depth": 12,
                 "n_heads": 12,
                 "qkv_bias": True,
                 "mlp_ratio": 4}

my_model = ViT(**custom_config)
my_model.eval()


for (n_o, p_o), (n_c, p_c) in zip(
        official_model.named_parameters(), my_model.named_parameters()
):
    assert p_o.numel() == p_c.numel()
    print(f"{n_o} | {n_c}")

    p_c.data[:] = p_o.data

    assert_tensors_equal(p_c.data, p_o.data)


"""
    Here we generate a random image with the required size to check
    our model's outputcompared to official pre-trained output.
"""
inp = torch.rand(1, 3, 384, 384)
my_output = my_model(inp)
official_output = official_model(inp)

# Asserts
assert get_n_params(my_model) == get_n_params(official_model)
# assert_tensors_equal(my_output, official_output)

torch.save(my_model, "./9_Transformers/ViT/saved_models/model.pth")

print("Verification is successful and the transferred model is saved!")