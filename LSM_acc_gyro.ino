#include <EdgeStreamWireless.h>
#include <SparkFunLSM6DSO.h>
#include <Wire.h>

LSM6DSO myIMU;
EdgeStreamWireless streamWireless;

const char* ssid = "Vedanta's Pixel";
const char* password = "hotspotpassword";

void setup() {
  // Start Serial communication for debugging
  Serial.begin(115200);

  Wire.begin();

  if( myIMU.begin() )
    Serial.println("Ready.");
  else { 
    Serial.println("Could not connect to IMU.");
    Serial.println("Freezing");
  }

  if( myIMU.initialize(BASIC_SETTINGS) )
    Serial.println("Loaded Settings.");

  // Initialize wireless data logging
  streamWireless.beginWireless(ssid, password);
}

void loop() {
  float acc_x = myIMU.readFloatAccelX();
  float acc_y = myIMU.readFloatAccelY();
  float acc_z = myIMU.readFloatAccelZ();

  float gyro_x = myIMU.readFloatGyroX();
  float gyro_y = myIMU.readFloatGyroY();
  float gyro_z = myIMU.readFloatGyroZ();

  // Show values on Serial Monitor
  Serial.print("Accel [X,Y,Z]: ");
  Serial.print(acc_x, 3); Serial.print(", ");
  Serial.print(acc_y, 3); Serial.print(", ");
  Serial.print(acc_z, 3);

  Serial.print(" | Gyro [X,Y,Z]: ");
  Serial.print(gyro_x, 3); Serial.print(", ");
  Serial.print(gyro_y, 3); Serial.print(", ");
  Serial.println(gyro_z, 3);

  const char* sensorName = "IMU";
  vector<double> sensorValues = {acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z};

  

  // Log the sensor data to the Serial Monitor
  streamWireless.logDataWireless(sensorName, sensorValues);

  // Delay of 1 second before the next loop iteration
  delay(500);
}