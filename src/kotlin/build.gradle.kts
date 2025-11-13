plugins {
    kotlin("jvm") version "2.1.10"
    application
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation("org.jetbrains.kotlin:kotlin-test")
    implementation("com.github.ajalt.clikt:clikt:4.2.0")
    implementation("com.github.samtools:htsjdk:4.0.1")
    implementation("org.biokotlin:biokotlin:1.0.0")
}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(19)
}
application{
    mainClass.set("SampleKt")
    applicationName = "sample"
}

tasks.jar {
    from(sourceSets.main.get().output)

}