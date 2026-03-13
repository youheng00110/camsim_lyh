waymo:
Processing sequences: 100%|██████████████████████████████████████████████████████████| 798/798 [00:00<00:00, 6796.59it/s]
Total valid windows generated: 10416
Binning windows: 100%|████████████████████████████████████████████████████████| 10416/10416 [00:00<00:00, 1276452.71it/s]
Sampling 150 windows per bin...
Sampling bins:   0%|                                                                              | 0/36 [00:00<?, ?it/s]Bin 0 (-180.0°): Found 503, Sampled 150
Bin 1 (-170.0°): Found 121, Sampled 121
Bin 2 (-160.0°): Found 172, Sampled 150
Bin 3 (-150.0°): Found 144, Sampled 144
Bin 4 (-140.0°): Found 199, Sampled 150
Bin 5 (-130.0°): Found 99, Sampled 99
Bin 6 (-120.0°): Found 77, Sampled 77
Bin 7 (-110.0°): Found 124, Sampled 124
Bin 8 (-100.0°): Found 1051, Sampled 150
Bin 9 (-90.0°): Found 402, Sampled 150
Bin 10 (-80.0°): Found 142, Sampled 142
Bin 11 (-70.0°): Found 182, Sampled 150
Bin 12 (-60.0°): Found 150, Sampled 150
Bin 13 (-50.0°): Found 207, Sampled 150
Bin 14 (-40.0°): Found 132, Sampled 132
Bin 15 (-30.0°): Found 91, Sampled 91
Bin 16 (-20.0°): Found 166, Sampled 150
Bin 17 (-10.0°): Found 1112, Sampled 150
Bin 18 (0.0°): Found 489, Sampled 150
Bin 19 (10.0°): Found 145, Sampled 145
Bin 20 (20.0°): Found 239, Sampled 150
Bin 21 (30.0°): Found 70, Sampled 70
Bin 22 (40.0°): Found 265, Sampled 150
Bin 23 (50.0°): Found 76, Sampled 76
Bin 24 (60.0°): Found 86, Sampled 86
Bin 25 (70.0°): Found 126, Sampled 126
Bin 26 (80.0°): Found 1361, Sampled 150
Bin 27 (90.0°): Found 421, Sampled 150
Bin 28 (100.0°): Found 92, Sampled 92
Bin 29 (110.0°): Found 119, Sampled 119
Bin 30 (120.0°): Found 165, Sampled 150
Bin 31 (130.0°): Found 168, Sampled 150
Bin 32 (140.0°): Found 116, Sampled 116
Bin 33 (150.0°): Found 113, Sampled 113
Bin 34 (160.0°): Found 114, Sampled 114
Bin 35 (170.0°): Found 1177, Sampled 150
Sampling bins: 100%|██████████████████████████████████████████████████████████████████| 36/36 [00:00<00:00, 20599.58it/s]

Final balanced dataset size: 4687 windows.

nuscenes

Train scenes used: 700
Generating windows and calculating motion angles...
Processing train scenes: 100%|████████████████████████████████████████████████████████| 700/700 [00:00<00:00, 702.13it/s]
Total valid windows generated: 9485
Binning windows...
100%|███████████████████████████████████████████████████████████████████████████| 9485/9485 [00:00<00:00, 1062579.42it/s]
Sampling 150 per bin...
  0%|                                                                                             | 0/36 [00:00<?, ?it/s]Bin 0: Found 167, Sampled 150
Bin 1: Found 148, Sampled 148
Bin 2: Found 330, Sampled 150
Bin 3: Found 463, Sampled 150
Bin 4: Found 397, Sampled 150
Bin 5: Found 95, Sampled 95
Bin 6: Found 52, Sampled 52
Bin 7: Found 70, Sampled 70
Bin 8: Found 489, Sampled 150
Bin 9: Found 157, Sampled 150
Bin 10: Found 122, Sampled 122
Bin 11: Found 266, Sampled 150
Bin 12: Found 979, Sampled 150
Bin 13: Found 521, Sampled 150
Bin 14: Found 113, Sampled 113
Bin 15: Found 96, Sampled 96
Bin 16: Found 85, Sampled 85
Bin 17: Found 158, Sampled 150
Bin 18: Found 90, Sampled 90
Bin 19: Found 158, Sampled 150
Bin 20: Found 443, Sampled 150
Bin 21: Found 481, Sampled 150
Bin 22: Found 322, Sampled 150
Bin 23: Found 69, Sampled 69
Bin 24: Found 97, Sampled 97
Bin 25: Found 84, Sampled 84
Bin 26: Found 478, Sampled 150
Bin 27: Found 265, Sampled 150
Bin 28: Found 207, Sampled 150
Bin 29: Found 568, Sampled 150
Bin 30: Found 573, Sampled 150
Bin 31: Found 425, Sampled 150
Bin 32: Found 155, Sampled 150
Bin 33: Found 97, Sampled 97
Bin 34: Found 89, Sampled 89
Bin 35: Found 176, Sampled 150
100%|█████████████████████████████████████████████████████████████████████████████████| 36/36 [00:00<00:00, 19732.74it/s]

Final balanced dataset size: 4607
AVrgo2

100%|█████████████████████████████████████████████████████████████████████████████████| 701/701 [00:04<00:00, 170.11it/s]
Total scenes: 700
Camera FPS: 20.0
Window size: 80
Step size: 20
Generating windows...
100%|███████████████████████████████████████████████████████████████████████████████| 700/700 [00:00<00:00, 16305.84it/s]
Total windows before balance: 7882
Saved all windows: ./avrgo2_balanced/all_windows.json
Sampling bins...
bin 0: 222 -> 150
bin 1: 155 -> 150
bin 2: 244 -> 150
bin 3: 106 -> 106
bin 4: 349 -> 150
bin 5: 120 -> 120
bin 6: 300 -> 150
bin 7: 351 -> 150
bin 8: 469 -> 150
bin 9: 226 -> 150
bin 10: 43 -> 43
bin 11: 83 -> 83
bin 12: 182 -> 150
bin 13: 98 -> 98
bin 14: 99 -> 99
bin 15: 156 -> 150
bin 16: 69 -> 69
bin 17: 504 -> 150
bin 18: 280 -> 150
bin 19: 126 -> 126
bin 20: 194 -> 150
bin 21: 164 -> 150
bin 22: 234 -> 150
bin 23: 56 -> 56
bin 24: 190 -> 150
bin 25: 352 -> 150
bin 26: 483 -> 150
bin 27: 268 -> 150
bin 28: 87 -> 87
bin 29: 177 -> 150
bin 30: 392 -> 150
bin 31: 113 -> 113
bin 32: 68 -> 68
bin 33: 172 -> 150
bin 34: 80 -> 80
bin 35: 670 -> 150
Final windows after balance: 4598

Nuplan


Loading nuPlan DB...
Total DB files: 64
Extracting ego metadata...                                                                                               
100%|████████████████████████████████████████████████████████████████████████████████████| 64/64 [03:20<00:00,  3.13s/it]
Total frames: 518999
Generating windows...
100%|███████████████████████████████████████████████████████████████████████████████████| 64/64 [00:00<00:00, 539.18it/s]
Total windows: 15725
Binning windows: 100%|████████████████████████████████████████████████████████| 15725/15725 [00:00<00:00, 1798130.60it/s]
Sampling windows...
100%|█████████████████████████████████████████████████████████████████████████████████| 36/36 [00:00<00:00, 22088.20it/s]
Balanced windows: 4817
Drawing polar comparison...
Done.