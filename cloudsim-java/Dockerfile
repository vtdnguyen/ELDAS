# ─────────────────────────────────────────────
# Stage 1: BUILD
# Dùng JDK đầy đủ + Maven để compile
# ─────────────────────────────────────────────
FROM eclipse-temurin:21-jdk-jammy AS builder

# Cài Maven (không có sẵn trong eclipse-temurin)
# Dùng version cụ thể để đảm bảo reproducible
ARG MAVEN_VERSION=3.9.9
RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && wget -q https://archive.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz \
    && tar -xzf apache-maven-${MAVEN_VERSION}-bin.tar.gz -C /opt \
    && ln -s /opt/apache-maven-${MAVEN_VERSION}/bin/mvn /usr/local/bin/mvn \
    && rm apache-maven-${MAVEN_VERSION}-bin.tar.gz \
    && apt-get purge -y wget && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# ── Trick quan trọng: copy pom.xml TRƯỚC source code ──
# Docker cache layer theo từng COPY instruction.
# Nếu pom.xml không đổi → layer "download dependency" được cache lại
# → lần build sau bỏ qua bước download (~2 phút tiết kiệm mỗi lần)
COPY pom.xml .
RUN mvn dependency:go-offline -B --no-transfer-progress

# Bây giờ mới copy source — layer này invalidate thường xuyên hơn
COPY src ./src

# Package thành fat JAR, bỏ qua test (test chạy riêng)
RUN mvn package -DskipTests -B --no-transfer-progress


# ─────────────────────────────────────────────
# Stage 2: RUNTIME
# Chỉ cần JRE (nhỏ hơn JDK ~200MB)
# ─────────────────────────────────────────────
FROM eclipse-temurin:21-jre-jammy AS runtime

# Tạo user riêng, không chạy process với quyền root
RUN groupadd --gid 1001 simuser \
    && useradd --uid 1001 --gid simuser --shell /bin/bash --create-home simuser

WORKDIR /app

# Chỉ copy file JAR từ stage builder — không có gì khác
COPY --from=builder /build/target/simulation-*.jar simulation.jar

# Tạo thư mục cho output, cấp quyền cho simuser
RUN mkdir -p /data/results /data/trace \
    && chown -R simuser:simuser /app /data

USER simuser

# Port Py4J Gateway — Python sẽ kết nối vào đây
EXPOSE 25333
# T5.3 — Prometheus metrics endpoint (chỉ phục vụ khi MONITORING_ENABLED=true).
# EXPOSE chỉ là documentation + dùng cho service discovery trong cùng network.
# Container-to-container scrape KHÔNG cần publish ra host.
EXPOSE 9091

# JVM tuning cho container:
# -XX:+UseContainerSupport  → JVM đọc giới hạn CPU/RAM từ cgroup thay vì host
# -XX:MaxRAMPercentage=75   → dùng tối đa 75% RAM được cấp phát cho container
# -XX:+ExitOnOutOfMemoryError → container tự restart thay vì treo im lặng
ENTRYPOINT ["java", \
    "-XX:+UseContainerSupport", \
    "-XX:MaxRAMPercentage=75.0", \
    "-XX:+ExitOnOutOfMemoryError", \
    "-jar", "simulation.jar"]