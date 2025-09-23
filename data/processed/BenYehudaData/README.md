```
Number of samples: 20123
100%|███████████████████████████████████| 20123/20123 [00:21<00:00, 922.36it/s]
Text lengths: min=157, max=2265972, mean=16915.90
Birth years: min=900, max=1929, mean=1657.35
Death years: min=970, max=2010, mean=1719.01
Number of unique authors: 1623
Total number of words: 58611132
Total number of characters: 340398712
```


Problem - Sometimes works are under translators instead of under authors - look at creation 6722 which is at '/p152/m6722' but created by 330 (152 as the translator)  
Fixed by indexing by author name

```
Number of samples: 20026
100%|██████████████████████████████████████████████| 20026/20026 [00:20<00:00, 959.76it/s]
Text lengths: min=157, max=2265972, mean=16314.92
Birth years: min=-460, max=1929, mean=1652.14
Death years: min=-395, max=2010, mean=1713.52
Number of unique authors: 1639
Total number of words: 56498486
```

Did some more fine tuning to the data:
```
Could not parse birth or death years for 93 authors
Number of samples: 20077
100%|█████████████████████████████████████████████| 20077/20077 [00:19<00:00, 1046.06it/s]
Text lengths: min=157, max=2265972, mean=16406.75
Birth years: min=-460, max=1929, mean=1650.08
Death years: min=-395, max=2010, mean=1711.49
Number of unique authors: 1658
Total number of words: 56988167
Total number of characters: 329398326
```

Also save plot to `Dist_1`

```
Loading Ben Yehuda texts: 21208it [00:03, 5916.88it/s]
Number of text samples: 20099

Text Content:
  Total characters: 329,491,055
  Average characters per text: 16,393

Composition Year Estimations Statistics:
  Range: -400 - 2010
  Mean: 1705
```