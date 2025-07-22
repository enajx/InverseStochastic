from torchvision.io import read_image
from torchvision.transforms.functional import resize
import numpy as np
import subprocess
from pathlib import Path
import time


def run_morpheus_blastocyst(params, xml_path, outdir, param_keys=None, supress_output=True):
    cmd = ["morpheus", "-f", xml_path, "--outdir", outdir]
    if params is not None:
        if isinstance(params, np.ndarray) and param_keys:
            for k, v in zip(param_keys, params):
                cmd += ["--set", f"{k}={v}"]
        elif isinstance(params, dict):
            for k, v in params.items():
                cmd += ["--set", f"{k}={v}"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL if supress_output else None)


def read_png(path, size=[224, 224], last=False):
    if last:
        path = Path(path)
        files = list(path.glob("plot_*.png"))
        if not files:
            raise FileNotFoundError("No matching PNG files found.")
        path = max(files, key=lambda p: int(p.stem.split("_")[1]))
        time.sleep(0.1)
        if path.stat().st_size == 0:
            raise ValueError(f"File {path} is empty.")
        # sleep for 1 second
        time.sleep(1)
        # print(path)
    img = read_image(path)
    img = resize(img, size)
    img = img.permute(1, 2, 0)  # 224, 224, 3 - uint8
    return img


if __name__ == "__main__":

    params = {
        "vsg1": 1.202,  # Maximum rate of Gata6 synthesis caused by ERK activation
        "vsg2": 1,  # Maximum rate of GATA6 synthesis caused by its auto-activation
        "vsn1": 0.856,  # Basal rate of NANOG synthesis
        "vsn2": 1,  # Maximum rate of NANOG synthesis caused by its auto-activation
        "vsfr1": 2.8,  # Basal rate of FGFR2
        "vsfr2": 2.8,  # Maximum rate of FGFR2 synthesis caused by GATA6 activation
        "vex": 0.0,  # Basal rate of FGF4 synthesis
        "vsf": 0.6,  # Maximum rate of FGF4 synthesis caused by NANOG activation
        "va": 20,  # ERK activation rate
        "vin": 3.3,  # ERK inactivation rate
        "kdg": 1,  # GATA6 degradation rate
        "kdn": 1,  # NANOG degradation rate
        "kdfr": 1,  # FGFR2 degradation rate
        "kdf": 0.09,  # FGF4 degradation rate
        "Kag1": 0.28,  # Threshold constant for the activation of GATA6 synthesis by ERK
        "Kag2": 0.55,  # Threshold constant for GATA6 auto-activation
        "Kan": 0.55,  # Threshold constant for NANOG auto-activation
        "Kafr": 0.5,  # Threshold constant for the activation of FGFR2 synthesis by GATA6
        "Kaf": 5,  # Threshold constant for the activation of FGF4 synthesis by NANOG
        "Kig": 2,  # Threshold constant for the inhibition of GATA6 synthesis by NANOG
        "Kin1": 0.28,  # Threshold constant for the inhibition of NANOG synthesis by ERK
        "Kin2": 2,  # Threshold constant for the inhibition of NANOG synthesis by GATA6
        "Kifr": 0.5,  # Threshold constant for the inhibition of FGFR2 synthesis by NANOG
        "Ka": 0.7,  # Michaelis constant for activation of the ERK pathway
        "Ki": 0.7,  # Michaelis constant for inactivation of the ERK pathway
        "Kd": 2,  # Michaelis constant for activation of the ERK pathway by FGF4
        "r": 3,  # Hill coefficient for the activation of GATA6 synthesis by ERK
        "s": 4,  # Hill coefficient for GATA6 auto-activation
        "q": 4,  # Hill coefficient for the inhibition of GATA6 synthesis by NANOG
        "u": 3,  # Hill coefficient for the inhibition of NANOG synthesis by ERK
        "v": 4,  # Hill coefficient for NANOG auto-activation
        "w": 4,  # Hill coefficient for the inhibition of NANOG synthesis by GATA6
        "z": 4,  # Hill coefficient for the activation of FGF4 synthesis by NANOG
        "k": 0.32,
        "b": 2,
        "a": 1,
        "th": 0.5,
        "i": 1.5,
        "d": 0.4,
        "n": 4,
        "basal": 0.0,
        "same": 0.0,
        "dcm": 0.0,
    }

    # run_morpheus_blastocyst(
    #     params=params,
    #     xml_path="src/models/Blastocyst/Mammalian_Embryo_Development.xml",
    #     outdir="temp_blastocyst",
    # )

    img = read_image("temp_blastocyst/plot_03000.png")
    img = resize(img, [224, 224])
    img = img.permute(1, 2, 0)  # 224, 224, 3 - uint8
    print(img.shape)
    print(img.dtype)

    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.show()
