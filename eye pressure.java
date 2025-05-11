import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class EyeBloodVesselDetection {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private JFrame frame;
    private JLabel liveLabel, captureLabel, processedLabel, pressureLabel;
    private VideoCapture camera;
    private Mat frameMat;

    public EyeBloodVesselDetection() {
        frame = new JFrame("Eye Blood Vessel Detection");
        frame.setLayout(new FlowLayout());
        frame.setSize(1000, 700);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        liveLabel = new JLabel();
        captureLabel = new JLabel();
        processedLabel = new JLabel();
        pressureLabel = new JLabel("Estimated Pressure: N/A");

        JButton btnLive = new JButton("Start Live View");
        JButton btnStop = new JButton("Stop Live View");
        JButton btnCapture = new JButton("Capture & Process");

        btnLive.addActionListener(e -> startCamera());
        btnStop.addActionListener(e -> stopCamera());
        btnCapture.addActionListener(e -> captureAndProcess());

        frame.add(liveLabel);
        frame.add(captureLabel);
        frame.add(processedLabel);
        frame.add(btnLive);
        frame.add(btnStop);
        frame.add(btnCapture);
        frame.add(pressureLabel);

        frame.setVisible(true);
    }

    private void startCamera() {
        camera = new VideoCapture(0);
        new Thread(() -> {
            frameMat = new Mat();
            while (camera.isOpened()) {
                camera.read(frameMat);
                if (!frameMat.empty()) {
                    ImageIcon image = new ImageIcon(matToBufferedImage(frameMat));
                    liveLabel.setIcon(image);
                    liveLabel.repaint();
                }
            }
        }).start();
    }

    private void stopCamera() {
        if (camera != null && camera.isOpened()) {
            camera.release();
        }
    }

    private void captureAndProcess() {
        if (frameMat != null && !frameMat.empty()) {
            Mat captured = frameMat.clone();
            captureLabel.setIcon(new ImageIcon(matToBufferedImage(captured)));

            // Convert to gray
            Mat gray = new Mat();
            Imgproc.cvtColor(captured, gray, Imgproc.COLOR_BGR2GRAY);

            // Histogram equalization
            Mat equalized = new Mat();
            Imgproc.equalizeHist(gray, equalized);

            // Edge detection
            Mat edges = new Mat();
            Imgproc.Canny(equalized, edges, 100, 200);

            // Morphology
            Mat morph = new Mat();
            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(4, 1));
            Imgproc.dilate(edges, morph, kernel);

            processedLabel.setIcon(new ImageIcon(matToBufferedImage(morph)));

            // Calculate metrics
            double length = Core.countNonZero(morph);
            double density = length / (morph.rows() * morph.cols());
            double pressure = calculatePressure(length, density);
            pressureLabel.setText(String.format("Estimated Pressure: %.2f mmHg", pressure));
        }
    }

    private double calculatePressure(double length, double density) {
        double original_pressure = (0.5 * length * density) / 10;
        double pressure = 10 + (original_pressure - 70) * (21 - 10) / (120 - 70);
        return Math.max(10, Math.min(21, pressure));
    }

    private BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) type = BufferedImage.TYPE_3BYTE_BGR;
        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] buffer = new byte[bufferSize];
        mat.get(0, 0, buffer);
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        final byte[] target = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(buffer, 0, target, 0, buffer.length);
        return image;
    }

    public static void main(String[] args) {
        new EyeBloodVesselDetection();
    }
}
