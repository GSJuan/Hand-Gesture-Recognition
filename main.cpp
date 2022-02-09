#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;

double angle(Point s, Point e, Point f) {

    double v1[2],v2[2];
    v1[0] = s.x - f.x;
    v1[1] = s.y - f.y;
    v2[0] = e.x - f.x;
    v2[1] = e.y - f.y;
    double ang1 = atan2(v1[1], v1[0]);
    double ang2 = atan2(v2[1], v2[0]);

    double ang = ang1 - ang2;
    if (ang > CV_PI) ang -= 2*CV_PI;
    if (ang < -CV_PI) ang += 2*CV_PI;

    return ang*180/CV_PI;
}

int main(int argc, char* argv[]) {     

        VideoCapture cap;
      
        bool record = false; //variable para, si es falsa indicar que no vfamos a grabar, en caso contrario, el programa servira para grabar videos      
        
        if (argc == 1) // no han pasado parametros, comportamiento por defecto
                cap.open(0); //detecta la webcam que haya conectada
        else{
                const string opcion = argv[1]; // comprobamos el parametro que se ha dado

                if(opcion == "-s") { // argumento de seleccion de fuente, le pasas el archivo de video a emplear
                        const string filename = argv[2];
                        cap.open(filename); 
                }

                else if(opcion == "-i") { // argumento de seleccion de fuente, le pasas la imagen a emplear
                        const string filename = argv[2];
                        //cap.open(filename); 
                }

                else if(opcion == "-c") { // parametro para grabar video, si se le pasa un nombre de archivo, asi nombrara la grabacion
                        record = true;
                        cap.open(0);
                }

                else if(opcion == "-h") { // parametro para mostrar la ayuda
                        cout << "Programa para reconocer gestos usando openCV, \n -h muestra la ayuda, \n -s nombre_archivo emplea como fuente un archivo de video, \n -c se ejecuta para grabar un video" << endl;
                        return 0;
                }
                else {
                        cout << "Opcion introducida no definida. Porfavor emplee la opcion -h para ver la ayuda" << endl;
                        return 1; // si es cualquier otro parámetro (indefinido) lo detecta y finaliza
                }
        }

        if (!cap.isOpened()) { // problemas para abrir la fuente de imagen, cierra el programa
                printf("Error opening cam\n");
                return -1;
        }   
    //existen dos posibilidades    
        if(record) { // si vamos a grabar un video

                Mat frame;

                //ventana donde se verá lo que se esté grabando
                namedWindow("Frame");

                //Variables de tamaño de imagen que debe tener el archivo final
                int frame_width = cap.get(CAP_PROP_FRAME_WIDTH); 
                int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT); 

                //nombre por defecto del archivo
                string filename = "default.avi";

                //establecemos el codec de salida
                int codec = VideoWriter::fourcc('M','J','P','G');

                if(argc == 3){ // si se le ha pasado el nombre del archivo de salida, le adjuntamos .avi al final
                        string temp = argv[2];
                        filename = temp;
                        filename += ".avi";
                }
                
                //objeto que permite escribir un archivo de video, le pasamos el nombre, el codec, los fps y el tamaño del ancho y alto
                VideoWriter video(filename,codec,20, Size(frame_width,frame_height)); 

                //bucle para grabar
                while(true) {
                        
                        //pasa los frames de la camara al objeto frame, de tipo Mat 
                        cap>>frame;

                        //muestra en la ventana lo que graba la cámara
                        imshow("Frame",frame);
                        
                        //graba en el archivo de video cada frame que grabe la camara
                        video.write(frame);

                        //para dejar de grabar, solo hay que pulsar q para salir
                        int c = waitKey(40);
                        if ((char)c =='q') break;
                }
        }
        else { // si vamos a procesar imagenes (camara o video)

                //declaramos las imagenes que vamos a procesar
                Mat frame, roi, fgMask, canvas;

                //declaramos las ventanas que apareceran
                namedWindow("Frame");
                namedWindow("ROI");
                namedWindow("ForeGround Mask");
                namedWindow("Canvas");

                //declaramos un rectangulo de 200x200 pixeles para marcar la región de interés(situada a 400 pixeles en el eje X y 100 en el Y de la esquina superior izquierda de la imagen)
                Rect rect(400,100,200,200);

                //variable que establece el learning rate del algoritmo que separa el fondo del frente de la imagen
                double learning_rate = -1.0;

                //creamos un puntero al algoritmo que vamos a emplear para separar el fondo
                Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2();

                //creamos un vector de vectores de puntos para almacenar los puntos que conforman el contorno de la mano
                vector<vector<Point>> contours;

                vector<Point> drawing;
                
                //bucle de procesado
                while(true) {

                        //las imagenes pasan frame a frame de la fuente a nuestro objeto de tipo Mat
                        cap>>frame;

                        //si en algun momento el frame es vacio, termina la ejecucion. Esto es para cuando la fuente es un archivo de video y este finaliza
                        if(frame.empty()) 
                                break;
                
                        //invertimos la imagen recibida para que sea mas facil de usar
                        flip(frame,frame,1);

                        //copiamos el area de interes del frame original a otro objeto de tipo Mat para procesar ese área
                        frame(rect).copyTo(roi);
                        //roi = frame(rect);

                        roi.copyTo(canvas);

                        //pintamos un rectangulo del tamaño exacto del área de interes en el frame original para mostrarlo en pantalla de color azul
                        rectangle(frame, rect, Scalar(255,0,0));

                        //Aplicamos el algoritmo para quitar el fondo a roi, y el resultado lo almacenamos en otro tipo Mat llamado fgMask, en el que se verá en funcionamiento el algoritmo.
                        //Además le indicamos el learning rate del algoritmo, para así poder variarlo en cualquier momento
                        pBackSub->apply(roi, fgMask, learning_rate);
                        
                        //recogemos la salida del algoritmo de extracción de fondo(Que se encuentra en fgMask) y encontramos el contorno de aquello distinto del fondo, almacenandolo en el vector de vectores de puntos
                        findContours(fgMask,contours,RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
                        
                        //variable para almacenar el índice del contorno más grande, asi evitamos pintar ruido generado por la camara
                        size_t max = 0;

                        //bucle para calcular la malla de convexion de cada elemento con contorno y para calcular el indice del mayor contorno
                        for(size_t i = 1; i < contours.size(); i++) {      
                                if(contours[i].size() > contours[max].size())
                                        max = i; 
                        }
                        
                        //convertimos el indice a un entero para poder usarlo
                        int biggest_contour_index = static_cast<int>(max);

                        //pinta por la pantalla roi los puntos del mayor contorno almacenado, de color verde y con un grosor de 2 pixeles
                        drawContours(roi, contours, biggest_contour_index, Scalar(0,255,0),2);

                        //vector que almacena la malla de convexidad
                        vector<int> hull;

                        //calculamos la malla de convexidad del mayor contorno detectado devolviendo los indices sobre el contorno
                        convexHull(contours[biggest_contour_index], hull, false, false);
                        
                        //ordenamos de mayor a menor el vector de la malla de convexidad
                        sort(hull.begin(),hull.end(), greater <int>());
                        
                        //vector para almacenar los defectos de convexidad
                        vector<Vec4i> defects;

                        //calculamos los defectos de convexidad en base al contorno y la malla de convexidad del mismo 
                        convexityDefects(contours[biggest_contour_index], hull, defects);
                        
                        //calculamos el bounding rect del mayor contorno
                        Rect bounding_rect = boundingRect(contours[biggest_contour_index]);

                        //pintamos el bounding rect en la ventana roi
                        rectangle(roi, bounding_rect,Scalar(0,0,255), 2);

                        // calculamos el momento del contorno de la mano
                        Moments m = moments(contours[biggest_contour_index]);

                        //creamos un punto con las coordenadas del centro de masa del contorno
                        Point cm(m.m10/m.m00, m.m01/m.m00);

                        //pintamos el punto por referencia
                        circle(roi, cm,5,Scalar(130,0,130),-1);

                        //contador de dedos en cada frame
                        int fingers  = 0;

                        //PUNTO MÁS ALEJADO DEL CENTRO, PUNTA DEL DEDO PARA PINTAR
                        Point furthest;

                        //mayor distancia entre el centroide y un defecto de convexidad
                        double max_distance = 0;

                        double distancia;

                        //bucle para procesar los defectos de convexidad
                        for (int i = 0; i < defects.size(); i++) {

                                //almacenamos los puntos start, end y furthest
                                Point s = contours[biggest_contour_index][defects[i][0]];
                                Point e = contours[biggest_contour_index][defects[i][1]];
                                Point f = contours[biggest_contour_index][defects[i][2]];

                                //calculamos la distancia del punto más lejano
                                float depth = (float)defects[i][3] / 256.0;
                               
                                //calculamos el ángulo entre las rectas que saldrian de unir  el punto f con el punto e y el punto s
                                double ang = angle(s,e,f);

                                //calculamos la distancia del punto s al centroide
                                double centroid_distance = sqrt(pow(s.x-cm.x, 2) + pow(s.y-cm.y,2));
                                
                                //si es mayor que la distancia máxima almacenada hasta el momento
                                if(centroid_distance > max_distance){
                                       
                                        //actualiza el valor
                                        max_distance = centroid_distance;

                                        //actualiza el punto mas lejano
                                        furthest = s;
                                }
                                
                                //filtramos los puntos de convexidad para pintar aquellos asociados a las uniones de los dedos (con una distancia mayor a 10 pixeles y con un ángulo menor a 95 grados)
                                if (ang < 90.0 && depth > 0.2 * bounding_rect.height) {

                                        //pintamos los puntos asociados a los defectos de convexidad ya filtrados
                                        circle(roi, f,5,Scalar(0,0,255),-1);

                                        //pintamos las lineas que unen los puntos e y s para formar la malla de convexion
                                        line(roi,s,e,Scalar(255,0,0),2);

                                        distancia = norm(s-e);

                                        //aumentamos en 1 la cantidad de dedos
                                        fingers ++;
                                }                            
                        }

                        if(fingers == 0 && bounding_rect.height < 1.35 * bounding_rect.width && bounding_rect.width < bounding_rect.height* 1.25) {

                          // si no hay defectos de convexidad y además el alto del bounding rect es similar al ancho
                                fingers = -1; // no hay ningun dedo levantado, la mano está cerrada, el bounding rect es casi un cuadrado 
                                
                                putText(frame, "Mano Cerrada", Point(30,85), FONT_HERSHEY_DUPLEX, 1.0, Scalar(0,0,255), 2, 5, false); //muestra el mensaje de que la mano esta cerrada   

                                drawing.clear();
                        }    
	                else if (fingers == 4) // si estan todos los dedos
		                putText(frame, "Mano Completamente Abierta", Point(30,85), FONT_HERSHEY_DUPLEX, 1.0, Scalar(0,255,0), 2, 5, false); //muestra el mensaje de que la mano esta totalmente abierta

                        else putText(frame, "Mano Parcialmente Abierta", Point(30,85), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255,0,0), 2, 5, false); // en otro caso, la mano esta parcialmente abierta
	                
                        //creamos la cadena que indicará el número de deos en cada momento
                        string dedos = "Dedos: ";
                        dedos += to_string(fingers+1);

                        //muestra en la imagen general el numero de dedos
                        putText(frame, dedos, Point(30,50), FONT_HERSHEY_PLAIN, 2.0, Scalar(0,0,0), 2, 5, false);

                        //si solo hay un dedo levantado, vamos a pintar
                        if(fingers == 0) {
                                //muestra el mensaje avisando de que estamos pintando
                                putText(frame, "Pintando!!", Point(30,115), FONT_HERSHEY_DUPLEX, 1.0, Scalar(155,155,0), 2, 5, false);

                                //almacenamos el punto asociado con el indice
                                drawing.push_back(furthest);

                                //recorremos el vector que almacena los puntos de lo pintado, y los mostramos todos para formar el dibujo
                                for(size_t i = 0; i < drawing.size(); i++) {
                                        circle(canvas, drawing[i],5,Scalar(155,0, 155),-1);
                                }
                        }
                        //si hay 2 dedos
                        else if(fingers == 1) {
                                //y la distancia entre ellos es mayor a 35 pixeles
                                if(distancia > 35.0)
                                        //reconocemos el simbolo del rock and roll
                                        putText(frame, "Rock & Roll!", Point(30,115), FONT_HERSHEY_DUPLEX, 1.0, Scalar(15,155,0), 2, 5, false);
                        }

                        //muestra en las ventanas abiertas el frame almacenado en cada objeto de tipo Mat
                        imshow("Frame",frame); //imagen original
                        imshow("ROI", roi); //contorno de la mano y malla de convexion
                        imshow("ForeGround Mask", fgMask); //mascara de fondo de la mano
                        imshow("Canvas", canvas); //imagen de lienzo

                        //para salir, pulsar la q, y para hacer que el algoritmo deje de aprender el fondo, pulsar la l de learning
                        int c = waitKey(40);
                        if ((char)c =='q') break;
                        else if((char)c == 'l') learning_rate = 0.0;
                }
                
                //libera los recursos
                cap.release();
                destroyAllWindows();
        }
}

