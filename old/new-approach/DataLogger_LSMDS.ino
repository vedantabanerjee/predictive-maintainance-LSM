#include <EdgeStream.h>
#include <SparkFunLSM6DSO.h>
#include <Wire.h>

LSM6DSO myIMU;
EdgeStream stream;  // wired (Serial) logging

void setup() {
  Serial.begin(115200);
  Wire.begin();

  if (myIMU.begin())
    Serial.println("IMU Ready.");
  else {
    Serial.println("Could not connect to IMU. Freezing...");
    while (1);
  }

  if (myIMU.initialize(BASIC_SETTINGS))
    Serial.println("Loaded IMU Settings.");

  Serial.println("EdgeStream initialized for wired logging (no begin() needed).");
}

void loop() {
  float acc_x = myIMU.readFloatAccelX();
  float acc_y = myIMU.readFloatAccelY();
  float acc_z = myIMU.readFloatAccelZ();

  float gyro_x = myIMU.readFloatGyroX();
  float gyro_y = myIMU.readFloatGyroY();
  float gyro_z = myIMU.readFloatGyroZ();

  const char* sensorName = "IMU";
  std::vector<double> sensorValues = {acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z};

  // Log over Serial using EdgeStream
  stream.logData(sensorName, sensorValues);

  delay(25);
}
