package sim;

public class Main {
    public static void main(String[] args) throws InterruptedException {
        System.out.println("CloudSim simulation container started.");
        // Giữ process sống để Docker không thoát ngay
        // Sau này sẽ thay bằng CloudSim + GatewayServer thật
        Thread.currentThread().join();
    }
}
