import org.gradle.kotlin.dsl.withType
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "2.1.21"
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
    implementation("com.google.guava:guava:33.1.0-jre")


}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(21)
}
tasks.withType<KotlinCompile> {
    compilerOptions.jvmTarget.set(JvmTarget.JVM_21)
}
application{
    mainClass.set("SampleKt")
    applicationName = "sample"
}

tasks.jar {
    from(sourceSets.main.get().output)

}