#define PinRelay1 7 //Vacuum motor
#define PinRelay2 6 //Solenoid valve

#define PinStep 10  //Step motor
#define PinDir 4
#define PinEn 3

#define IN1 8       //DC motor - L298N
#define IN2 5

String coordinate = "";
float distance_offset = 69;           // Distance from home position to the edge of the frame (mm)
float distance_to_target = 0;         // Distance from the edge of the frame to the button (mm)
float distance_resolution = 40;       // Distance traveled when step completes one revolution (mm) 
float pulse_resolution = 200;         // One revolution needs 200 pulses(mm)
float pixel_to_mm = 0.2844311;        // 1 px = 0.284 mm (mm/pxl) 
float pixel = 0;                      
int pulse = 0;                        

void setup() 
{
  Serial.begin(9600);

 pinMode(IN1, OUTPUT); 
 pinMode(IN2, OUTPUT);
  
  pinMode(PinRelay1, OUTPUT); 
  pinMode(PinRelay2, OUTPUT); 

  pinMode(PinDir, OUTPUT);
  pinMode(PinStep, OUTPUT);
  pinMode(PinEn, OUTPUT);

  digitalWrite(PinRelay1, LOW);
  digitalWrite(PinRelay2, LOW);

  digitalWrite(PinEn, LOW);

  conveyor_on();
}

void conveyor_on()                       // on conveyor
{
 digitalWrite(IN1, HIGH);
 analogWrite(IN2, 127);
}

 void conveyor_off()                   // off conveyor
 {
  digitalWrite(IN1, LOW);
  analogWrite(IN2, LOW);
 }

void calculate_distance()
{
    distance_to_target = pixel_to_mm * pixel;                                                 // convert pixel to mm
    pulse = (distance_offset + distance_to_target) * pulse_resolution / distance_resolution;  // convert mm to pulse 
}

void grab_object() 
{
  // run step
  digitalWrite(PinDir, HIGH);
  for (int i = 0; i < pulse; i++) 
  {
    digitalWrite(PinStep, HIGH);
    delayMicroseconds(1000); 
    digitalWrite(PinStep, LOW);
    delayMicroseconds(1000);
  }

  delay(10400);                          // timming 

  conveyor_off()

  digitalWrite(PinRelay1, HIGH);        // on vacuum 
    
  digitalWrite(PinRelay2, HIGH);        // lower cylinder
  delay(500);
  digitalWrite(PinRelay2, LOW);         // upper cylinder
  delay(500);

  conveyor_on();
  
  // run inverted step
  digitalWrite(PinDir, LOW);
  for (int i = pulse; i > 0; i--) 
  {
    digitalWrite(PinStep, HIGH);
    delayMicroseconds(1000); 
    digitalWrite(PinStep, LOW);
    delayMicroseconds(1000);
  }
  
  
  digitalWrite(PinRelay1, LOW);         // off vacuum

  digitalWrite(PinRelay2, HIGH);        // lower cylinder
  delay(500);
  digitalWrite(PinRelay2, LOW);         // upper cylinder
  delay(500);

  Serial.print("done");
}

void loop() 
{
  if (Serial.available() > 0) {
    coordinate = Serial.readStringUntil('\n');
    pixel = coordinate.toInt(); 

    calculate_distance();
    Serial.println(distance_offset + distance_to_target);
    grab_object();
  }
}
