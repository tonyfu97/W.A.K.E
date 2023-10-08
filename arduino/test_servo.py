"""
Use this program to test the servo motors.

>>> python3 test_servo.py

Then enter the numbers (0-6) when prompted to test the servo motors.

"""

import serial
import time

arduino = serial.Serial(port = '/dev/cu.usbmodem1101', timeout=0)
time.sleep(2)

while True:

    print("Enter the channel A intensity: ")
    A = str(input())
    
    if (A == "q"):
        exit()
    
    print("Enter the channel B intensity: ")
    B = str(input())
    
    arduino.write(str.encode(f"{A}, {B}"))
    print(f"Enter 'q' to quit")
    
    