# Dataset files

The dataset is not included in this repository.

The Colab notebook expects these archives in:

```text
/content/drive/MyDrive/Bachelor's Project/
```

| Archive | Extracted file | Expected array shape |
| --- | --- | --- |
| `data_27_cases_30_minutes.zip` | `X.npy` | `(27, 460800, 19)` |
| `data_33_cases_30_minutes.zip` | `data_33_cases_30_minutes.npy` | `(33, 460800, 19)` |

Each subject contains 30 minutes of 19-channel EEG sampled at 256 Hz.

The difference between the first archive name and its internal `X.npy` filename is
inherited from the original project files.

