"""
Batch size optimization utilities for CTLearn predictions.
This module provides functions to test and find optimal batch sizes for model inference,
helping to maximize GPU utilization while avoiding out-of-memory errors.
"""

import torch


def test_batch(model, imgs, peak_time, device):
    """
    Test if a model can process a given batch without memory errors.
    
    This function tests whether a pre-prepared batch of images and peak times
    can be successfully processed by the model without encountering out-of-memory
    (OOM) errors. It's used to validate batch sizes during optimization.
    
    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to test.
    imgs : torch.Tensor or array-like
        Batch of images to process. Will be converted to tensor if necessary.
    peak_time : torch.Tensor or array-like
        Batch of peak time information. Will be converted to tensor if necessary.
    device : torch.device or str
        Device to run the test on (e.g., 'cuda:0' or 'cpu').
    
    Returns
    -------
    bool
        True if the batch can be processed successfully, False if OOM error occurs.
    
    Raises
    ------
    RuntimeError
        If a RuntimeError other than OOM occurs during processing.
    
    Notes
    -----
    The function automatically clears the CUDA cache after each test to ensure
    clean memory state for subsequent tests.
    """
    # Move model to specified device and set to evaluation mode
    model.to(device)
    model.eval()

    # Ensure inputs are tensors and move to device
    if not torch.is_tensor(imgs):
        imgs = torch.as_tensor(imgs).to(device)
    else:
        imgs = imgs.to(device)

    if not torch.is_tensor(peak_time):
        peak_time = torch.as_tensor(peak_time).to(device)
    else:
        peak_time = peak_time.to(device)

    try:
        # Attempt forward pass without gradient computation
        with torch.no_grad():
            _ = model(imgs, peak_time)
        
        # Clear CUDA cache to free memory
        torch.cuda.empty_cache()
        return True

    except RuntimeError as e:
        # Check if error is due to out of memory
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False
        else:
            # Re-raise unexpected errors after cleaning up
            torch.cuda.empty_cache()
            raise e


def find_max_batch_size(self, model, imgs, peak_time, device, start_bs=8, step=8, max_bs=512):
    """
    Find the maximum batch size that can be processed without OOM errors.
    
    This function performs a binary-like search to find the largest batch size
    that can be successfully processed by the model on the given device. It starts
    with a small batch size and incrementally increases until an OOM error occurs.
    
    Parameters
    ----------
    self : object
        Reference to the parent object (for potential logging or configuration access).
    model : torch.nn.Module
        The PyTorch model to test.
    imgs : torch.Tensor or array-like
        Sample images to use for testing. Only the first image is used and replicated.
    peak_time : torch.Tensor or array-like
        Sample peak time data. Only the first value is used and replicated.
    device : torch.device or str
        Device to run tests on (e.g., 'cuda:0' or 'cpu').
    start_bs : int, optional
        Initial batch size to start testing with. Default is 8.
    step : int, optional
        Increment step for batch size increases. Default is 8.
    max_bs : int, optional
        Maximum batch size to test. Default is 512.
    
    Returns
    -------
    int
        The maximum batch size that can be processed without OOM errors.
    
    Raises
    ------
    RuntimeError
        If a RuntimeError other than OOM occurs during testing.
    
    Notes
    -----
    - The function replicates a single image/peak_time to create test batches
    - CUDA cache is cleared after each test to ensure accurate memory measurements
    - Progress is printed to console with emoji indicators for status
    """
    batch_size = start_bs
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()

    # Ensure inputs are tensors
    if not torch.is_tensor(imgs):
        imgs = torch.as_tensor(imgs)
    if not torch.is_tensor(peak_time):
        peak_time = torch.as_tensor(peak_time)

    # Disable gradient computation for efficiency
    with torch.no_grad():
        while batch_size <= max_bs:
            try:
                # Prepare image batch
                batch_imgs = imgs[:1]  # Take first image as template
                if batch_imgs.ndim == 3:
                    batch_imgs = batch_imgs.unsqueeze(1)  # Add channel dimension if needed
                # Replicate to create batch of desired size
                batch_imgs = batch_imgs.repeat(batch_size, 1, 1, 1).to(device)

                # Prepare peak_time batch
                value = peak_time[:1, 0, 0]  # Extract representative value
                batch_peaks = value.unsqueeze(1)  # Shape [1, 1]
                batch_peaks = batch_peaks.repeat(batch_size, 1)  # Replicate for batch
                batch_peaks = batch_peaks.unsqueeze(-1).unsqueeze(-1).to(device)  # Shape [batch, 1, 1, 1]

                # Attempt forward pass
                _ = model(batch_imgs, batch_peaks)

                # Clean up tensors and cache
                del batch_imgs, batch_peaks, _
                torch.cuda.empty_cache()

                print(f"✅ Batch size {batch_size} OK")
                batch_size += step

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # OOM encountered, return previous successful batch size
                    print(f"💥 OOM at batch size {batch_size}")
                    torch.cuda.empty_cache()
                    return batch_size - step
                else:
                    # Unexpected error, clean up and re-raise
                    print(f"❌ Unexpected error at batch size {batch_size}: {e}")
                    torch.cuda.empty_cache()
                    raise e

    # If we reached max_bs without OOM, return it (minus step to be safe)
    return batch_size - step
