import numpy as np


def positional_encoding(pos, d_model, max_len=10):
    """
    Calculate positional encodings using the formula:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
    - pos: The position for which to calculate the encoding.
    - d_model: The dimension of the model.
    - max_len: The maximum sequence length for which to calculate encodings.

    Returns:
    - A 1D numpy array of shape (d_model,) containing the positional encoding for the given position.
    """
    # Create an array of divisors for each dimension of the encoding
    divisors = np.power(10000, 2 * np.arange(d_model) / d_model)

    # Calculate the sine and cosine components of the encoding
    sin_component = np.sin(pos / divisors)
    cos_component = np.cos(pos / divisors)

    # Interleave the sine and cosine components
    encoding = np.stack([sin_component, cos_component], axis=-1)

    return encoding


if __name__ == "__main__":
    pos = 1
    d_model = 20
    encoding = positional_encoding(pos, d_model)
    print(encoding)
