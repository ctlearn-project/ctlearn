import torch

def test_batch(model, imgs, peak_time, device):
    """
    Tests a batch already prepared with imgs and peak_time.
    Returns True if the model can process it without errors, False if there is an OOM (Out of Memory) error.
    """
    model.to(device)
    model.eval()

    if not torch.is_tensor(imgs):
        imgs = torch.as_tensor(imgs).to(device)
    else:
        imgs = imgs.to(device)

    if not torch.is_tensor(peak_time):
        peak_time = torch.as_tensor(peak_time).to(device)
    else:
        peak_time = peak_time.to(device)

    try:
        with torch.no_grad():
            _ = model(imgs, peak_time)
        torch.cuda.empty_cache()
        return True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False
        else:
            torch.cuda.empty_cache()
            raise e


def find_max_batch_size(self, model, imgs, peak_time, device, start_bs=8, step=8, max_bs=512):
    batch_size = start_bs
    model.to(device)
    model.eval()

    if not torch.is_tensor(imgs):
        imgs = torch.as_tensor(imgs)
    if not torch.is_tensor(peak_time):
        peak_time = torch.as_tensor(peak_time)

    with torch.no_grad():
        while batch_size <= max_bs:
            try:
                # Images
                batch_imgs = imgs[:1]
                if batch_imgs.ndim == 3:
                    batch_imgs = batch_imgs.unsqueeze(1)  # Add channel
                batch_imgs = batch_imgs.repeat(batch_size, 1, 1, 1).to(device)

                # peak_time
                value = peak_time[:1, 0, 0]          # Take representative value
                batch_peaks = value.unsqueeze(1)     # Shape [1,1]
                batch_peaks = batch_peaks.repeat(batch_size, 1)
                batch_peaks = batch_peaks.unsqueeze(-1).unsqueeze(-1).to(device)  # Shape [batch,1,1,1]

                # Forward
                _ = model(batch_imgs, batch_peaks)

                del batch_imgs, batch_peaks, _
                torch.cuda.empty_cache()

                print(f"✅ Batch size {batch_size} OK")
                batch_size += step

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"💥 OOM at batch size {batch_size}")
                    torch.cuda.empty_cache()
                    return batch_size - step
                else:
                    print(f"❌ Unexpected error at batch size {batch_size}: {e}")
                    torch.cuda.empty_cache()
                    raise e

    return batch_size - step
