void setup() {
  //PC모니터로 센서값을 확인하기위해서 시리얼 통신을 정의 
  Serial.begin(9600);
}

void loop()
{
  //초음파 출력
  pinMode(10, OUTPUT);
  digitalWrite(10, LOW);
  delayMicroseconds(2);
  digitalWrite(10, HIGH);
  delayMicroseconds(10);
  digitalWrite(10, LOW);
  //초음파 입력
  pinMode(9, INPUT);
  double duration = pulseIn (9, HIGH);
  double distance = duration * 17 / 1000; 

  //PC모니터로 초음파 거리값을 확인 하는 코드
  Serial.println(duration ); //초음파가 반사되어 돌아오는 시간

  Serial.print("\nDIstance : ");

  Serial.print(distance); //측정된 물체로부터 cm값

  Serial.println(" Cm");


  //delay(1000); //1초마다 측정값
  

  //led출력하는 핀번호
  pinMode(7, OUTPUT); //빨간색
  pinMode(6, OUTPUT); //초록색
  pinMode(5, OUTPUT); //파란색
  
  //led 출력 코드
  if (distance <= 100){
    digitalWrite(7, HIGH);
    digitalWrite(6, LOW);
    digitalWrite(5, LOW);
    tone(4, 2093,800);
      delay(500);
  }
  else if (distance <=200){
    digitalWrite(7, LOW);
    digitalWrite(6, HIGH);
    digitalWrite(5, LOW);
    tone(4, 2093,800);
      delay(1000);
  }
  else {
    digitalWrite(7, LOW);
    digitalWrite(6, LOW);
    digitalWrite(5, HIGH);
  }
}
