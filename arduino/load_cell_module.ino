#include <HX711_ADC.h> // need to install 
#include <Wire.h>
#include <LiquidCrystal_I2C.h> // need to install
#include <SoftwareSerial.h>  

//pins:
const int HX711_dout = 3; //mcu > HX711 dout pin
const int HX711_sck = 2; //mcu > HX711 sck pin
const int BT_TX = 12;
const int BT_RX = 13;

SoftwareSerial BTSerial(BT_TX,BT_RX);        // 소프트웨어 시리얼 객체를 12(TX),13번(RX) 으로 생성
HX711_ADC LoadCell(HX711_dout, HX711_sck); // parameters: dt pin 3, sck pin 2;
LiquidCrystal_I2C lcd(0x27, 16,2); // 0x27 is the i2c address might different;you can check with Scanner

void setup() 
{
  Serial.begin(9600);                // 시리얼 통신을 개시, 속도는 9600  
  BTSerial.begin(9600);              // 소프트웨어시리얼 통신 개시, 속도는 9600
  LoadCell.begin(); // start connection to HX711
  LoadCell.start(2000); // load cells gets 2000ms of time to stabilize
  LoadCell.setCalFactor(234.0); // calibration factor for load cell => dependent on your individual setup
  lcd.init(); 
  lcd.backlight();
}

void loop() {
  if(Serial.available())              // 시리얼 통신으로 문자가 들어오면
  {
    BTSerial.write(Serial.read());      // 블루투스시리얼 통신으로 발송
    // lcd.print("BTSerial : "); // print out to LCD
    // lcd.print(Serial.read());
  }
  if(BTSerial.available())               // 블루투스 시리얼 통신으로 문자가 들어오면
  {
    Serial.write(BTSerial.read());       // 시리얼 창에 표시(시리얼 통신으로 출력)
    // lcd.print("Serial : "); // print out to LCD
    // lcd.print(BTSerial.read());
  }
  
  LoadCell.update(); // retrieves data from the load cell
  float i = LoadCell.getData(); // get output value


  lcd.setCursor(0, 0); // set cursor to first row
  lcd.print("Weight[g]:"); // print out to LCD
  // BTSerial.write("Weight[g]:");
  lcd.setCursor(0, 1); // set cursor to second row
  lcd.print(i); // print out the retrieved value to the second row


  // Bluetooth Serial 출력 (수정된 부분)
  // 1. float을 문자열로 변환
  char buffer[16]; // 충분한 크기의 버퍼
  dtostrf(i, 6, 2, buffer); // float -> string 변환 (전체 6자리, 소수점 이하 2자리)

  Serial.println("Weight : ");
  Serial.println(buffer);
  
  // 2. 문자열 전송
  BTSerial.print(buffer);
  BTSerial.print("\n"); // 줄바꿈 추가 (선택사항)
  // BTSerial.write(i);
  delay(200);

}