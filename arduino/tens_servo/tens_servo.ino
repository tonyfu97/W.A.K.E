/*
 * Set two servo motors to the desired degree positions.
 * Used to control the intensities the TENS units.
 */

#include <Servo.h>

#define A_PIN 9
#define B_PIN 6
#define MAX_DEGREE 175
#define MAX_DIAL 6
#define NUM_STEPS 20

Servo servoA;
Servo servoB;

void setup()
{
  Serial.begin(9600);
  servoA.attach(A_PIN);
  servoB.attach(B_PIN);
  servoA.write(0);
  servoB.write(0);
}

void loop()
{
  static float degreeA = 0;
  static float degreeB = 0;
  while (Serial.available())
  {
    // Pulse width and rate have range [0, MAX_DAIL], need to map it to [0, MAX_DEGREE] degrees.
    degreeA = map(Serial.parseInt(), 0, MAX_DIAL, 0, MAX_DEGREE);
    degreeB = map(Serial.parseInt(), 0, MAX_DIAL, 0, MAX_DEGREE);

    // Clamp the degrees
    degreeA = (degreeA > MAX_DEGREE) ? MAX_DEGREE : degreeA;
    degreeB = (degreeB > MAX_DEGREE) ? MAX_DEGREE : degreeB;

    // For debug
    Serial.print(degreeA);
    Serial.print(',');
    Serial.println(degreeB);

    // We want to slowly turn the dial from the previous value.
    static float prevDegreeA = 0;
    static float prevDegreeB = 0;
    for (int i = 0; i < NUM_STEPS; i++)
    {
      float tempDegreeA = (1 - i / float(NUM_STEPS)) * prevDegreeA + (i / float(NUM_STEPS)) * degreeA;
      float tempDegreeB = (1 - i / float(NUM_STEPS)) * prevDegreeB + (i / float(NUM_STEPS)) * degreeB;
      servoA.write((int)tempDegreeA);
      servoB.write((int)tempDegreeB);
      delay(30);
    }

    Serial.parseInt(); // Flush the trailing \0 character from the serial buffer
    delay(15);

    prevDegreeA = degreeA;
    prevDegreeB = degreeB;
  }
}
