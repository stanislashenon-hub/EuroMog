// Cal Hacks 13.0 - Uno firmware: MPU6050 streaming + motor PWM + Hall tachometer
//
// Wiring (Uno R3):
//   MPU6050  VCC->5V  GND->GND  SDA->A4  SCL->A5
//   L293D    pin1 (EN)->D9  pin2 (IN1)->D7  pin7 (IN2)->D8
//            pin16 (VCC1)->5V  pin8 (VCC2)->motor battery +
//            pins 4,5 -> GND   pins 3,6 -> motor
//   Hall     VCC->5V  GND->GND  OUT->D2   (magnet taped on rotor)
//   All grounds connected together (Uno, L293D, motor battery -).
//
// Output, 1000 lines/s at 500000 baud:  ax,ay,az,rpm   (accel in milli-g)
// Input commands (send from laptop, end with newline):
//   P<0-255>   set motor PWM duty, e.g. P150
//   S          stop motor

#include <Wire.h>

const int EN = 9, IN1 = 7, IN2 = 8, TACH = 2;
const uint8_t MPU = 0x68;
const unsigned long PERIOD_US = 1000;     // 1 kHz sampling

volatile unsigned long lastPulse = 0, period = 0;
unsigned long nextSample;
char cmd[8];
uint8_t cmdLen = 0;

void onPulse() {
  unsigned long now = micros();
  if (now - lastPulse > 2000) {           // ignore glitches (<2 ms apart)
    period = now - lastPulse;
    lastPulse = now;
  }
}

void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

void setup() {
  Serial.begin(500000);
  Wire.begin();
  Wire.setClock(400000);                  // fast I2C
  mpuWrite(0x6B, 0x00);                   // wake up
  mpuWrite(0x1A, 0x01);                   // low-pass filter ~184 Hz, 1 kHz internal rate
  mpuWrite(0x1C, 0x08);                   // accel range +/-4 g (8192 counts per g)

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  digitalWrite(IN1, HIGH);                // one direction only
  digitalWrite(IN2, LOW);
  analogWrite(EN, 0);                     // motor off at start

  pinMode(TACH, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(TACH), onPulse, FALLING);

  nextSample = micros();
}

void readCommands() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      cmd[cmdLen] = 0;
      if (cmd[0] == 'P') analogWrite(EN, constrain(atoi(cmd + 1), 0, 255));
      if (cmd[0] == 'S') analogWrite(EN, 0);
      cmdLen = 0;
    } else if (cmdLen < sizeof(cmd) - 1) {
      cmd[cmdLen++] = c;
    }
  }
}

void loop() {
  readCommands();

  if ((long)(micros() - nextSample) < 0) return;   // wait for next sample time
  nextSample += PERIOD_US;
  if ((long)(micros() - nextSample) > 10000) nextSample = micros();  // fell far behind: resync

  Wire.beginTransmission(MPU);
  Wire.write(0x3B);                       // ACCEL_XOUT_H
  Wire.endTransmission(false);
  Wire.requestFrom(MPU, (uint8_t)6);
  int16_t ax = (Wire.read() << 8) | Wire.read();
  int16_t ay = (Wire.read() << 8) | Wire.read();
  int16_t az = (Wire.read() << 8) | Wire.read();

  noInterrupts();
  unsigned long p = period, t = lastPulse;
  interrupts();
  long rpm = (p > 0 && micros() - t < 500000) ? 60000000UL / p : 0;

  // integers in milli-g: much faster to print than floats on the Uno
  Serial.print((long)ax * 1000 / 8192); Serial.print(',');
  Serial.print((long)ay * 1000 / 8192); Serial.print(',');
  Serial.print((long)az * 1000 / 8192); Serial.print(',');
  Serial.println(rpm);
}
