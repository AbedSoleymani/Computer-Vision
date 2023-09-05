import torch

"""
Defining the device parameter as a global parameter across
different moduls by:
    import device
at the begining of each modul and having the global variable as:
    device.device
"""
device = "mps" if torch.backends.mps.is_available() else "cpu"