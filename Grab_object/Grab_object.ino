#define PinRelay1 7 //dc bom hut
#define PinRelay2 6 //van khi nen - xylanh

#define PinStep 10
#define PinDir 4
#define PinEn 3

#define IN1 8
#define IN2 5

String coordinate = "";
float distance_offset = 69;           // k/c tu vi tri home den ria khung anh (mm)
float distance_to_target = 0;         // k/c tu ria khung anh den vi tri cuc ao (mm)
float distance_resolution = 40;       // k/c di duoc khi quay 1 vong dong co step (mm) 
float pulse_resolution = 200;         // 1 vong quay step can 200 xung (mm)
float pixel_to_mm = 0.2844311;          // ty le pixel so voi k/c (mm/pxl) 
float pixel = 0;                      // toa do x_c cua cuc ao tren anh (pxl)
int pulse = 0;                        // gia tri xung de dieu khien dong co step

void setup() 
{
  Serial.begin(9600);

 pinMode(IN1, OUTPUT); 
 pinMode(IN2, OUTPUT);
  
  pinMode(PinRelay1, OUTPUT); // bom hut
  pinMode(PinRelay2, OUTPUT); // van khi nen

  pinMode(PinDir, OUTPUT);
  pinMode(PinStep, OUTPUT);
  pinMode(PinEn, OUTPUT);

  digitalWrite(PinRelay1, LOW);
  digitalWrite(PinRelay2, LOW);

  digitalWrite(PinEn, LOW);

  conveyor_on();
}

void conveyor_on()                    // chay bang tai
{
 digitalWrite(IN1, HIGH);
 analogWrite(IN2, 127);
}

// void conveyor_off()                   // dung bang tai
// {
//  digitalWrite(IN1, LOW);
//  analogWrite(IN2, LOW);
// }

void calculate_distance()
{
    distance_to_target = pixel_to_mm * pixel;                                                 // chuyen pixel sang mm
    pulse = (distance_offset + distance_to_target) * pulse_resolution / distance_resolution;  // chuyen mm sang xung 
}

void grab_object() 
{
  // chay step thuan
  digitalWrite(PinDir, HIGH);
  for (int i = 0; i < pulse; i++) 
  {
    digitalWrite(PinStep, HIGH);
    delayMicroseconds(1000); 
    digitalWrite(PinStep, LOW);
    delayMicroseconds(1000);
  }

  delay(10500);                          // thơi gian doi vat den vị trí hút

  //conveyor_off()

  digitalWrite(PinRelay1, HIGH);        // bat bom hut
    
  digitalWrite(PinRelay2, HIGH);        // ha xi lanh
  delay(100);
  digitalWrite(PinRelay2, LOW);         // nang xilanh
  delay(100);
  
  // chay step nghich
  digitalWrite(PinDir, LOW);
  for (int i = pulse; i > 0; i--) 
  {
    digitalWrite(PinStep, HIGH);
    delayMicroseconds(1000); 
    digitalWrite(PinStep, LOW);
    delayMicroseconds(1000);
  }
  
  // conveyor_on();
  digitalWrite(PinRelay1, LOW);         // tat bom hut

  digitalWrite(PinRelay2, HIGH);        // ha xi lanh
  delay(500);
  digitalWrite(PinRelay2, LOW);         // nang xilanh
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
