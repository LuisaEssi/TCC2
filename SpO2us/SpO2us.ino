#include <splash.h>
#include <Adafruit_SSD1306.h>

#define LED_IR 32

#define LED_RED 33

#define FOTODIODO 34

#define ordem 32

#define VALOR_R 255
#define VALOR_IR 255

unsigned char controle_LEDs = 0;/* Responsavel pela comutacao do acionamento dos LEDs*/

int m = 5, cont = 0, compensa = 0, indice = 0, i =0;
float optical_red = 0, leitura_luam = 0, optical_ir = 0, sinal_optical_red = 0, sinal_optical_ir = 0;
float media_optical_red = 0, media_optical_ir = 0, leitura_red = 0, leitura_ir = 0, leitura_ir_comp = 0, leitura_red_comp = 0, leitura_red_i = 0, leitura_ir_i = 0;

double media_movel_red[ordem], media_movel_ir[ordem], saida_fil_ir = 0, saida_fil_red = 0;

///INICIO
 
void setup() 
{
  
  Serial.begin(115200);
  pinMode(LED_IR, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  pinMode(FOTODIODO, INPUT);
  
//  dacWrite(CANAL_DAC0, 0);  
//  dacWrite(CANAL_DAC1, 0);
  
  
  digitalWrite(LED_RED,HIGH);
  digitalWrite(LED_IR,HIGH); 
  
for (int i = 0; i < ordem; i++) { 
  media_movel_red[i] = 0; 
  media_movel_ir[i]=0; 
  
  }  
}
 
void loop() 
{       
media_optical_red = 0;
media_optical_ir = 0;
//cont = cont + 1;

//     -------------- Liga Vermelho -----------------

     
             digitalWrite(LED_RED,LOW);
             digitalWrite(LED_IR,HIGH);            

             delayMicroseconds(105);
//             delay(1000);             
             media_optical_red = 0;
             leitura_red = 0;
             leitura_red_i = 0;
             optical_red = 0;
             leitura_red_i= analogRead(FOTODIODO);
             for(i=0; i<m; i++){
                 leitura_red += analogRead(FOTODIODO);         //Conversao AD sinal vermelho
                 
                  }  
                         
             media_optical_red = leitura_red/m;
             
             sinal_optical_red += (media_optical_red); 
             sinal_optical_red -= media_movel_red[indice];
             
             media_movel_red[indice] = (media_optical_red);
             
             saida_fil_red = (sinal_optical_red)/ordem;
           
             delayMicroseconds(115);
//             delay(1000);

//      --------------- Desliga ----------------

             digitalWrite(LED_RED,HIGH);
             digitalWrite(LED_IR,HIGH);
             
             delayMicroseconds(160);
//             delay(1000);
             leitura_luam = 0;
             leitura_luam = analogRead (FOTODIODO);    //Conversao AD iluminacao ambiente
             delayMicroseconds(160);
//             delay(1000);
                
//     ------------ Liga Infravermelho -------------

   
             digitalWrite(LED_RED,HIGH);
             digitalWrite(LED_IR, LOW);
  
             delayMicroseconds(105);
//             delay(1000);
             media_optical_ir = 0; 
             leitura_ir = 0;
             leitura_ir_i = 0;
             optical_ir = 0;
             leitura_ir_i= analogRead(FOTODIODO);      
             for(i=0; i<m; i++){
               leitura_ir += analogRead(FOTODIODO);         //Conversao AD sinal infravermelho
               
                }      
             media_optical_ir = leitura_ir/m;

             sinal_optical_ir += media_optical_ir; 
             sinal_optical_ir -= media_movel_ir[indice];
             media_movel_ir[indice] = media_optical_ir;
                                       
             saida_fil_ir = (sinal_optical_ir)/ordem;
             
             delayMicroseconds(115);
//             delay(1000);
            
//      --------------- Desliga ----------------
              

             digitalWrite(LED_RED,HIGH);
             digitalWrite(LED_IR,HIGH);
//             delay(1000);
             delayMicroseconds(1193);  
             
//       ----------- Converter para valor de tensao ----------

             
//             saida_fil_red = (saida_fil_red*3.3)/4095;
//             saida_fil_ir = (saida_fil_ir*3.3)/4095;
////
////             optical_red = (optical_red*3.3)/4095;
////             optical_ir = (optical_ir*3.3)/4095;
//
//             leitura_red_i = (leitura_red_i*3.3)/4095;
//             leitura_ir_i= (leitura_ir_i*3.3)/4095;

             saida_fil_red = 3.3 - ((saida_fil_red*3.3)/4095);
             saida_fil_ir = 3.3 - ((saida_fil_ir*3.3)/4095);

             leitura_red_i =  3.3 - ((leitura_red_i*3.3)/4095);
             leitura_ir_i= 3.3 - ((leitura_ir_i*3.3)/4095);

             
//        ----------- Printar dados --------

             Serial.print(saida_fil_red,6);
             Serial.print(",");
             Serial.print(saida_fil_ir,6);
             Serial.print(",");
             Serial.print(leitura_red_i,6);
             Serial.print(",");
             Serial.println(leitura_ir_i,6);
//             Serial.print(",");
//             Serial.println(leitura_luam,6);
         

              indice++;
              indice %= ordem;
}
