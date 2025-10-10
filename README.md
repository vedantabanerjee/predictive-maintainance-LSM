# predictive-maintainance-LSM

### hardware
- LSMDSOX
- ESP32C3

date: 28th September 2025
- dataset was made having 9,000 samples in each class (3k samples from each point). Sample rate is unknown and potentially ambiguous.
- motor_on(class_label=0), motor_off(class_label=1), motor_on_nofan(class_label=2) was made

date: october 10th 2025
- data was taken and a new dataset was made with 18,000 samples in each class (6k samples from each point). Sample Rate was ~32samples/second => 40Hz.
- motor_on(class_label=0), motor_off(class_label=1), motor_on_nofan(class_label=2) and motor_on_badfan(class_label=3) was made.
- for class 3, one side of the existing blade was broken to make it unbalanced. 