// define LEFT and RIGHT pin number
// left side: pin 8, 9, 10 (use)
// right side: 2, 3 (use), 4
#define LEFT 10
#define RIGHT 3

#define pinport PIOC
#define pinmask (1<<21)
// This script is to deliver vibro-tactile stimulation for moudulating brain activity around motor cortex
// In the current setting (2021.07.12), left and right index finger will be placed on left and right motor, respectively.
// Pipeline overview:
// s1. Matlab code extracts features from segmented EEG
// s2. When the features match the pre-defined condition, it sends commands (left or right) to Arudino through Serial(9600)
// s3. Accoriding to the commands, appropriate side motor will vibrate for pre-defined duration
// s4. Send "end-command" to Matlab code not to receive overlapped commands during vibration
// Script: Kyungho Won (khwon.public@gmail.com)

// ------------------------------ Vars --------------------- //
bool isRunning = false; // for preventing overlapped command while already running
int VT_duration_ms = 100;

void vibrate_on_dur_power(int pinnum, int duration, float intensity){
  // analogWrite -> intensity: 0 to 255
  analogWrite(pinnum, intensity*3.6/5.0);
  delay(duration);
  analogWrite(pinnum, 0);
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600); // desktop <-> Arduino
  // Unity -> Arduino (Analog): intensity (0 to N)
  // Arduino -> motor (Analog): intensity (0 to N)-> G
  
  // right 3 motors -> currently only 3 is used
  pinMode(2, OUTPUT); // PWM #2
  pinMode(3, OUTPUT); // PWM #3
  pinMode(4, OUTPUT); // PWM #4
  
  // left 3 motors -> currently only 10 is used
  pinMode(8, OUTPUT); // PWM #8
  pinMode(9, OUTPUT); // PWM #9
  pinMode(10, OUTPUT); // PWM #10
}

void loop() {
  // put your main code here, to run repeatedly:
  // to do: activate the actuator when the requirment is met.
  
  // get stimulation flag from Unity

    if (Serial.available() > 0)
    {
      char commands_from_MI = Serial.read(); // vibrating left or right
      int m_speed = commands_from_MI - '0'; // expected value: 1 or 2
      //Serial.write(speed); // rollback for debug
        if (isRunning == false)
        {
            switch(m_speed)
            {
              // for now less than 100ms is not enough to generate meaningful vibration intensity to attend
              case 1:
                isRunning = true;
                vibrate_on_dur_power(LEFT, VT_duration_ms, 255); // was 2.1
                isRunning = false;
                break;
              case 2:
                isRunning = true;
                vibrate_on_dur_power(RIGHT, VT_duration_ms, 255); // was 3.6
                isRunning = false;
                break;
              default:
                break;
            }
            // Serial.write('e'); // end of stimulation  
          } 
  }
  

}
