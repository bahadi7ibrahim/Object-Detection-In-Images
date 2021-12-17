package sample;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.stage.FileChooser;

import java.awt.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.opencv.core.Core;
import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.dnn.Net;
import org.opencv.utils.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.io.FileOutputStream;
import java.util.ListIterator;

public class Controller {
    @FXML
    public Button btn1, btn2;
    public Label label;

    public String imagePath;

    private Desktop desktop = Desktop.getDesktop();

    //function return the output image name
    public static String getRandomStr() {

        String str = "1234567890";

        StringBuilder s = new StringBuilder(16);
        s.append("ENSAH-151220-");
        for (int i = 0; i < 3; i++) {
            int index = (int) (str.length() * Math.random());
            s.append(str.charAt(index));
        }
        s.append(".png");
        return s.toString();
    }

    private static List<String> getOutputNames(Net net) {
        List<String> names = new ArrayList<>();
        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();

        outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));
        return names;
    }

    public void ButtonSelectAction(ActionEvent event) {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setInitialDirectory(new File("C:\\Users\\Redouan\\Pictures\\"));
        fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("JPG Image", "*.jpg"),
                new FileChooser.ExtensionFilter("PNG Image", "*.png"),
                new FileChooser.ExtensionFilter("JPEG Image", "*.jpeg")
        );
        File selectedImage = fileChooser.showOpenDialog(null);

        if (selectedImage != null) {
            imagePath = selectedImage.getAbsolutePath();
            label.setText(imagePath);
            System.out.println(label);
        } else {
            System.out.println("Image is not valid!");
        }
    }

    public void ButtonDetectAction(ActionEvent event) throws Exception {
        if(label.getText() != null) {
            String img_objectsDetected = detectObjects(label.getText());
            openFile(new File(img_objectsDetected));
            System.out.println("success");
        }else {
            System.out.println("Ops!! there was a problem.");
        }
    }

    public static String detectObjects(String path)  throws Exception {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String modelWeights = "C:\\Users\\Redouan\\Documents\\yolov3.weights";
        String modelConfiguration = "C:\\Users\\Redouan\\Documents\\yolov3.cfg";
        String modelNames = "C:\\Users\\Redouan\\Documents\\coco.names";


        ArrayList<String> classes = new ArrayList<>();
        FileReader file = new FileReader(modelNames);
        BufferedReader bufferedReader = new BufferedReader(file);
        String Line;
        while ((Line = bufferedReader.readLine()) != null) {
            classes.add(Line);
        }
        bufferedReader.close();

        Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights); //OpenCV DNN supports models trained from various frameworks like Caffe and TensorFlow. It also supports various networks architectures based on YOLO//

        //object localisation components
        Mat image = Imgcodecs.imread(path);
        Size sz = new Size(416, 416);
        Mat blob = Dnn.blobFromImage(image, 0.00392, sz, new Scalar(0), true, false);
        System.out.println(blob);
        net.setInput(blob);

        java.util.List<Mat> result = new ArrayList<>();
        java.util.List<String> outBlobNames = getOutputNames(net);
        net.forward(result, outBlobNames);
        
        // Minimum probability to filter out weak detections. I gave a default value of 50%.
        float confThreshold = 0.3f;

        LinkedList<Integer> clsIds = new LinkedList<>();
        java.util.List<Float> confs = new ArrayList<>();
        List<Rect> rects = new ArrayList<>();

        for (int i = 0; i < result.size(); i++)
        {

            // each row is a candidate detection, the 1st 4 numbers are
            // [center_x, center_y, width, height], followed by (N-4) class probabilities

            Mat level = result.get(i);
            for (int j = 0; j < level.rows(); ++j)
            {
                Mat row = level.row(j);
                Mat scores = row.colRange(5, level.cols());

                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float)mm.maxVal;
                org.opencv.core.Point classIdPoint = mm.maxLoc;

                if (confidence > confThreshold)
                {

                    int centerX = (int)(row.get(0,0)[0] * image.cols());
                    int centerY = (int)(row.get(0,1)[0] * image.rows());
                    int width   = (int)(row.get(0,2)[0] * image.cols());
                    int height  = (int)(row.get(0,3)[0] * image.rows());
                    int left    = centerX - width  / 2;
                    int top     = centerY - height / 2;

                    clsIds.addLast((int)classIdPoint.x);
                    confs.add(confidence);
                    System.out.println(confidence);
                    rects.add(new Rect(left, top, width, height));

                }
            }
        }
        
        //This is our non-maximum suppression threshold with a default value of 0.3
        float nmsThresh = 0.3f;

        MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));

        Rect[] boxesArray = rects.toArray(new Rect[0]);

        MatOfRect boxes = new MatOfRect(boxesArray);
        MatOfInt indices = new MatOfInt();

        Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);

        // Draw result boxes:
        int [] ind = indices.toArray();

        for (int i = 0; i < ind.length; ++i)
        {
            int idx = ind[i];
            Rect box = boxesArray[idx];
            Imgproc.rectangle(image, box.tl(), box.br(), new Scalar(255,255,0), 5,20);

            String label = classes.get(clsIds.get(i)).toString();
            Imgproc.putText(image,label,new Point(box.x+30,box.y), 2, 2, new Scalar(0,0,255),2);

            // System.out.println(box);
        }
        String resultname = getRandomStr();
        Imgcodecs.imwrite(resultname, image);
        return resultname;
    }

    private void openFile(File file) {
        try {
            desktop.open(file);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(
                    Level.SEVERE, null, ex
            );
        }
    }
}
