//----------------------------------------------------------------------------------------------------------------------------------------------
//-   초음파 센서로 측정된 거리가 10cm 이하일 경우 
//- /home/pi/Pictures디렉토리에 사진 저장
//----------------------------------------------------------------------------------------------------------------------------------------------

//- 라이브러리 포함 --------------------------------------------------------------------------------------------------------
#include <stdio.h>
#include <wiringPi.h>


//- 연결 센서 핀 번호 선언 ------------------------------------------------------------------------------------------
#define   TRIG_PIN       3		    
#define   ECHO_PIN       23    

//- 사용자 정의 함수 선언 -------------------------------------------------------------------------------------------
int  getDistance();

//- 전역변수 선언 ------------------------------------------------------------------------------------------------------------
unsigned long  preMillis = 0;
char  cmd[] = "sudo raspistill -o /home/pi/Pictures/test.jpg";

//----------------------------------------------------------------------------------------------------------------------------------------------
//- Entry Point 함수 
//----------------------------------------------------------------------------------------------------------------------------------------------
int main(void)
{
	//- wiringPi 초기화 ------------------------------------------------------------------
	if(wiringPiSetup()==-1)
	{
			printf("wiringPi setup fail.");
			return 0;
	}
	
	//- 센서 연결 핀 동작모드 및 초기화 -------------------------------------------
	pinMode(TRIG_PIN, OUTPUT);
	pinMode(ECHO_PIN, INPUT);
	
	//- 기능 구현 --------------------------------------------------------------------------------------------
	while(1)
	{
		   if(millis() - preMillis>=500)
		   {
					int dis = getDistance();
					printf("Distance = %dcm\n", dis);
					
					if( dis <= 10)
					{
						system(cmd);		// 10cm 이하로 접근 시 사진  촬영
					}
					preMillis = millis();
		   }
		   
	}
	
	return 0;
}

//- 초음파 센서를 활용한 거리 (cm)  구하기 ------------------------------------------------------------------------------------
int  getDistance(){
	unsigned int  duration = 0,  start_time = 0, end_time = 0;
	float distance = 0;
	
	
	   //- TRIG PIN으로 초음파 발사 명령 신호 출
		digitalWrite(TRIG_PIN, LOW);
		delay(500);
		
		digitalWrite(TRIG_PIN, HIGH);
		delayMicroseconds(10);
		digitalWrite(TRIG_PIN, LOW);
		
		//- ECHO PIN으로 반사되어 돌아오는 초음파 계산
		while( digitalRead(ECHO_PIN) == LOW)
			start_time = micros();
		
		while( digitalRead(ECHO_PIN) == HIGH)
			end_time = micros();
		
		//- 시간 -> 거리로 변환
		duration = (end_time - start_time)/2;
		distance = 340 * ((float)duration/1000000) *100; 

	    //- cm 거리 변환 후 리턴
	     return (int)distance;	
}
