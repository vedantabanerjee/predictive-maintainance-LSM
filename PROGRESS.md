# On-Device Machine Learning for Vibration-Based Predictive Maintenance of Industrial Induction Motors Using MEMS Sensors

### hardware
- SmartElex 6 Degrees of Freedom - LSM6DSOX
- SeedStudio ESP32C3

### motor specs
- type: capacitor start & run single phase induction motor
- power input: 0.37KW (0.5 HP)
- volts: 230v - 50Hz
- RPM - 2800

### progress update

date: november 7th 2025
- New dataset is collected which is now sampled at avg. 450 samples/sec ~ 450Hz. This should solve the problem of aliasing
- The taining dataset now contains 50k samples for each class and the test dataset has 10k samples for each class
- Till now all models perform really well in testing but very bad in Inference.

date: october 15th 2025
- EDA notebook is uploaded where there is comprehensive EDA is performed on the new extensive dataaset
- take1 notebook is uploaded with a simple model on the new dataset giving an accuracy of 94% is uploaded.

date: october 13th 2025
- new additional data was taken and a new dataset with 36,000 samples in each class (12k samples from each point).
- Sample rate is ~ 32samples/second => 40Hz.
- dataset now contains 144k sample data points.

date: october 10th 2025
- data was taken and a new dataset was made with 18,000 samples in each class (6k samples from each point). Sample Rate was ~32samples/second => 40Hz.
- motor_on(class_label=0), motor_off(class_label=1), motor_on_nofan(class_label=2) and motor_on_badfan(class_label=3) was made.
- for class 3, one side of the existing blade was broken to make it unbalanced. 

date: 28th September 2025
- dataset was made having 9,000 samples in each class (3k samples from each point). Sample rate is unknown and potentially ambiguous.
- motor_on(class_label=0), motor_off(class_label=1), motor_on_nofan(class_label=2) was made