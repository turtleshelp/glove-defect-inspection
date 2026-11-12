/*
  Glove sorter – 3-position servo diverter (PASS / REWORK / REJECT)

  Hardware: Arduino Uno/Nano + one hobby servo (e.g. SG90/MG90S)
    servo signal -> D9, servo V+ -> 5 V (external 5 V supply recommended), GND shared with Arduino
  Serial: 9600 baud, one command per line from the PC application:
    'P' -> PASS bin, 'R' -> REWORK bin, 'X' -> REJECT bin, 'H' -> home (PASS)
  Reply:  "OK <cmd> <angle>"  or  "ERR <char>"

  Angles must match config/config.yaml (sorter.angles).
*/
#include <Servo.h>

const int SERVO_PIN = 9;
const int ANGLE_PASS = 30;
const int ANGLE_REWORK = 90;
const int ANGLE_REJECT = 150;
const unsigned long HOLD_MS = 600;   // keep the flap in position while the glove passes

Servo flap;

void moveTo(int angle) {
  flap.write(angle);
}

void setup() {
  Serial.begin(9600);
  flap.attach(SERVO_PIN);
  moveTo(ANGLE_PASS);
  Serial.println("READY glove_sorter");
}

void loop() {
  if (!Serial.available()) return;
  char c = Serial.read();
  if (c == '\n' || c == '\r' || c == ' ') return;
  int angle;
  switch (c) {
    case 'P': case 'H': angle = ANGLE_PASS; break;
    case 'R': angle = ANGLE_REWORK; break;
    case 'X': angle = ANGLE_REJECT; break;
    default:
      Serial.print("ERR ");
      Serial.println(c);
      return;
  }
  moveTo(angle);
  Serial.print("OK ");
  Serial.print(c);
  Serial.print(' ');
  Serial.println(angle);
  delay(HOLD_MS);
}
