#include <splash.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_GFX.h>
#include <SPI.h>
#include <Wire.h>
#include <SPIFFS.h>
#include <esp_task_wdt.h>
#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels

#define BLACK           0x0000

#define LED_IR 32

#define LED_RED 33

#define FOTODIODO 34

#define ordem 32

#define OLED_MOSI   23
#define OLED_CLK    18
#define OLED_DC     16
#define OLED_CS     5
#define OLED_RESET  17

int leituraAtual = 1;
#define posicao_x 5
#define posicao_y 40
#define altura 20
#define comprimento 110

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT,
  OLED_MOSI, OLED_CLK, OLED_DC, OLED_RESET, OLED_CS);

#define NUMFLAKES     10 // Number of snowflakes in the animation example

#define LOGO_HEIGHT   16
#define LOGO_WIDTH    16

static const unsigned char PROGMEM logo_bmp[] =
{ 0b00000000, 0b11000000,
  0b00000001, 0b11000000,
  0b00000001, 0b11000000,
  0b00000011, 0b11100000,
  0b11110011, 0b11100000,
  0b11111110, 0b11111000,
  0b01111110, 0b11111111,
  0b00110011, 0b10011111,
  0b00011111, 0b11111100,
  0b00001101, 0b01110000,
  0b00011011, 0b10100000,
  0b00111111, 0b11100000,
  0b00111111, 0b11110000,
  0b01111100, 0b11110000,
  0b01110000, 0b01110000,
  0b00000000, 0b00110000 };

char c;
char str[50] = "";
char string_red[30] = "";
char string_ir[30] = "";
uint8_t idx = 0;
int counter_linha = 0;
unsigned char controle_LEDs = 0;/* Responsavel pela comutacao do acionamento dos LEDs*/

int m = 5, cont = 0, compensa = 0, indice = 0, i =0, saida_fil_ir_A = 0, saida_fil_red_A = 0;
float optical_red = 0, leitura_luam = 0, optical_ir = 0, sinal_optical_red = 0, sinal_optical_ir = 0;
float media_optical_red = 0, media_optical_ir = 0, leitura_red = 0, leitura_ir = 0, leitura_ir_comp = 0, leitura_red_comp = 0, leitura_red_i = 0, leitura_ir_i = 0;

double media_movel_red[ordem], media_movel_ir[ordem], saida_fil_ir = 0, saida_fil_red = 0;

char buf[]="Arduino";

//variaveis que indicam o núcleo
static uint8_t taskCoreZero = 0;
static uint8_t taskCoreOne  = 1;

//FUNCOES

void comutaLEDs( void * pvParameters ) {
   
  while(true){
    esp_task_wdt_init(30, false); 
    media_optical_red = 0;
    media_optical_ir = 0;
    
    //     -------------- Liga Vermelho -----------------
    
    digitalWrite(LED_RED,LOW);
    digitalWrite(LED_IR,HIGH);            
    
    delayMicroseconds(105); 
               
    media_optical_red = 0;
    leitura_red = 0;
    leitura_red_i = 0;
    optical_red = 0;
//    leitura_red_i= analogRead(FOTODIODO);
    for(i=0; i<m; i++){
      leitura_red += analogRead(FOTODIODO);         //Conversao AD sinal vermelho
    }  
             
    media_optical_red = leitura_red/m;    
//    sinal_optical_red += (media_optical_red); 
//    sinal_optical_red -= media_movel_red[0];    
//    media_movel_red[indice] = (media_optical_red);    
//    saida_fil_red = sinal_optical_red/ordem;
    
    delayMicroseconds(115);
     
    //      --------------- Desliga ----------------
    
    digitalWrite(LED_RED,HIGH);
    digitalWrite(LED_IR,HIGH);    
    delayMicroseconds(160);
    leitura_luam = 0;
    leitura_luam = analogRead (FOTODIODO);    //Conversao AD iluminacao ambiente
    delayMicroseconds(160);
     
    
    //     ------------ Liga Infravermelho -------------
    
    
    digitalWrite(LED_RED,HIGH);
    digitalWrite(LED_IR, LOW);    
    delayMicroseconds(105);
     
    media_optical_ir = 0; 
    leitura_ir = 0;
    leitura_ir_i = 0;
    optical_ir = 0;
//    leitura_ir_i= analogRead(FOTODIODO);      
    for(i=0; i<m; i++){
      leitura_ir += analogRead(FOTODIODO);        //Conversao AD sinal infravermelho    
    }    
      
    media_optical_ir = leitura_ir/m;  
//    sinal_optical_ir += media_optical_ir; 
//    sinal_optical_ir -= media_movel_ir[0];
//    media_movel_ir[indice] = media_optical_ir;                       
//    saida_fil_ir = sinal_optical_ir/ordem;
//    
    delayMicroseconds(115);
//
//    Serial.println(saida_fil_ir);
//     
    
    //      --------------- Desliga ----------------
    
    digitalWrite(LED_RED,HIGH);
    digitalWrite(LED_IR,HIGH);
    delayMicroseconds(1240);  

    indice ++;
    indice %= ordem;
    
    //       ----------- Converter para valor de tensao ----------
    saida_fil_red_A = media_optical_red;
    saida_fil_ir_A = media_optical_ir;
    
    saida_fil_red = 3.3 - ((media_optical_red*3.3)/4095);
    saida_fil_ir = 3.3 - ((media_optical_ir*3.3)/4095);
    
//    leitura_red =  3.3 - ((leitura_red_i*3.3)/4095);
//    leitura_ir= 3.3 - ((leitura_ir_i*3.3)/4095);

    
    
    //        ----------- Printar dados filtrados --------
    
    Serial.print(saida_fil_red_A);
    Serial.print(",");
    Serial.println(saida_fil_ir_A);

     
    
    }
}


void LigaOLED( void * pvParameters ){
   
  while(true){
    esp_task_wdt_init(30, false); 
//    if(Serial.available() > 0){ // There's a command    
//      c = Serial.read(); // Read one byte
//       
//    
//      if(c != '\n'){ // Still reading
//        str[idx++] = c; // Parse the string byte (char) by byte
//        
//      }
//      else{ // Done reading      
//        str[idx] = '\0'; // Convert it to a string
//        display.setTextColor(WHITE);
//        display.setCursor(0,4);
//        display.setTextSize(2);
//        display.clearDisplay();
//        display.println(str);
//        display.display();
//        idx = 0;
//      }
//     
//    }else{
//      str[idx] = '\0'; // Convert it to a string
//      display.setTextColor(WHITE);
//      display.setCursor(0,34);
//      display.setTextSize(2);
//      display.display();  
//    }   

    int grafico_red = map(saida_fil_red_A, 750, 850, 0, 20);

    display.drawPixel(posicao_x+leituraAtual, altura-grafico_red, WHITE);
    display.setTextSize(2);
    display.display();  
    
    leituraAtual++;

    if(leituraAtual == 110) {
      //limpa a área toda do gráfico
      display.fillRect(posicao_x+1,posicao_y-1, comprimento, altura-1, BLACK);
      leituraAtual = 1; //volta o contador de leitura para 1 (nova coordenada X)   
      display.clearDisplay();
   }
   
//   delay(100);
  }  
}



void setup() {
  
  Serial.begin(115200);
   
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);  
  display.fillScreen(BLACK);

  // Clear the buffer.
  display.clearDisplay();  
  // Display Text
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0,28);
  display.println("Começa aqui!");
  display.display();
  delay(2000);
  
  display.clearDisplay();   
  pinMode(LED_IR, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  pinMode(FOTODIODO, INPUT);  
  
  digitalWrite(LED_RED,HIGH);
  digitalWrite(LED_IR,HIGH); 
  
  for (int i = 0; i < ordem; i++) { 
    media_movel_red[i] = 0; 
    media_movel_ir[i]=0;     
  }  

  xTaskCreatePinnedToCore(
                  comutaLEDs,   /* função que implementa a tarefa */
                  "comutaLEDs", /* nome da tarefa */
                  10000,      /* número de palavras a serem alocadas para uso com a pilha da tarefa */
                  NULL,       /* parâmetro de entrada para a tarefa (pode ser NULL) */
                  3,          /* prioridade da tarefa (0 a N) */
                  NULL,       /* referência para a tarefa (pode ser NULL) */
                  taskCoreZero);         /* Núcleo que executará a tarefa */
                  
  delay(10); //tempo para a tarefa iniciar

  xTaskCreatePinnedToCore(
                  LigaOLED,   /* função que implementa a tarefa */
                  "LigaOLED", /* nome da tarefa */
                  10000,      /* número de palavras a serem alocadas para uso com a pilha da tarefa */
                  NULL,       /* parâmetro de entrada para a tarefa (pode ser NULL) */
                  1,          /* prioridade da tarefa (0 a N) */
                  NULL,       /* referência para a tarefa (pode ser NULL) */
                  taskCoreOne);         /* Núcleo que executará a tarefa */
                  
  delay(50); //tempo para a tarefa iniciar
//  
}


void loop(){
  }
