// onset_analyzer.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <aubio/aubio.h>
#include <pthread.h>
#include <cjson/cJSON.h>

// Define export macro for potential future DLL exports; here it's not strictly needed.
#ifdef _WIN32
  #define API declspec(dllexport)
#else
  #define API __attribute((visibility("default")))
#endif

// Maximum number of detected onset events (adjust if needed)
#define MAX_EVENTS 10000

// Structure to hold information about each detected onset event
typedef struct {
    double time;            // Smoothed onset time in seconds
    double pitch;           // Detected pitch in Hz
    double predicted_next;  // Predicted time for the next onset (seconds)
} OnsetEvent;

// Global array to store onset events and a counter
static OnsetEvent events[MAX_EVENTS];
static int event_count = 0;

// Mutex for thread-safe access to events array
pthread_mutex_t event_mutex = PTHREAD_MUTEX_INITIALIZER;

/*
 * Simple 1D Kalman filter for smoothing time measurements.
 * It estimates the true onset time by reducing measurement noise.
 */
double kalman_filter(double measurement) {
    // static variables keep their values between function calls
    static double x = 0.0; // must be a constant expression
    static double P = 1.0;
    static int initialized = 0;

    // If this is the first call, initialize x with the current measurement
    if (!initialized) {
        x = measurement;
        initialized = 1;
    }

    double Q = 1e-4; // process noise
    double R = 1e-2; // measurement noise

    double K = P / (P + R);  // Kalman gain
    x = x + K * (measurement - x); // update estimate
    P = (1.0 - K) * P + Q;         // update error covariance

    return x;
}

/*
 * Simple AR(1) model to predict the next onset time.
 * If at least one previous event exists, it predicts the next onset
 * based on the interval between the last two events; otherwise, returns the current time.
 */
double predict_next_onset(double last_time, double previous_time) {
    double interval = last_time - previous_time;
    double phi = 0.9;  // AR(1) coefficient (tune as needed)
    double predicted_interval = phi * interval;
    return last_time + predicted_interval;
}

// Structure to pass parameters to the analysis thread
typedef struct {
    const char *filepath;  // Path to the WAV file
    int track_index;       // Track index (for multi-track files; here we process as mono)
} TrackParams;

/*
 * Thread function to process one audio track.
 * It uses Aubio to detect onsets and pitch from the WAV file.
 */
void* process_track(void *arg) {
    TrackParams *params = (TrackParams*) arg;
    const char *filepath = params->filepath;
    int track_index = params->track_index;  // Not used in this simple example (processes mono)

    // Analysis parameters
    uint_t hop_size = 512;
    uint_t win_size = 1024;
    uint_t samplerate = 0;  // Will be set from the file

    // Create an Aubio source object to read the WAV file
    aubio_source_t *source = new_aubio_source(filepath, samplerate, hop_size);
    if (!source) {
        fprintf(stderr, "Error: Cannot open file %s\n", filepath);
        pthread_exit(NULL);
    }
    samplerate = aubio_source_get_samplerate(source);

    // Create an onset detector (using default method)
    aubio_onset_t *onset = new_aubio_onset("default", win_size, hop_size, samplerate);
    if (!onset) {
        fprintf(stderr, "Error: Cannot create onset detector\n");
        del_aubio_source(source);
        pthread_exit(NULL);
    }
    aubio_onset_set_threshold(onset, 0.3f);
    aubio_onset_set_silence(onset, -40.f);

    // Create a pitch detector (using default method)
    aubio_pitch_t *pitch = new_aubio_pitch("default", win_size, hop_size, samplerate);
    if (!pitch) {
        fprintf(stderr, "Error: Cannot create pitch detector\n");
        del_aubio_onset(onset);
        del_aubio_source(source);
        pthread_exit(NULL);
    }
    aubio_pitch_set_unit(pitch, "Hz");
    aubio_pitch_set_silence(pitch, -40.f);
    // Allocate audio buffers
    fvec_t *in_buffer = new_fvec(hop_size);
    fvec_t *onset_output = new_fvec(1);
    fvec_t *pitch_output = new_fvec(1);
    if (!in_buffer || !onset_output || !pitch_output) {
        fprintf(stderr, "Error: Cannot allocate buffers\n");
        if (in_buffer) del_fvec(in_buffer);
        if (onset_output) del_fvec(onset_output);
        if (pitch_output) del_fvec(pitch_output);
        del_aubio_pitch(pitch);
        del_aubio_onset(onset);
        del_aubio_source(source);
        pthread_exit(NULL);
    }

    uint_t read = 0;
    double previous_smoothed_time = 0.0;

    // Main processing loop: read blocks until end of file
    while (1) {
        aubio_source_do(source, in_buffer, &read);
        if (read == 0) break;  // End of file reached

        // Process onset detection
        aubio_onset_do(onset, in_buffer, onset_output);
        float onset_flag = fvec_get_sample(onset_output, 0);

        // Process pitch detection
        aubio_pitch_do(pitch, in_buffer, pitch_output);
        float current_pitch = fvec_get_sample(pitch_output, 0);

        // If an onset is detected in this block
        if (onset_flag >= 1.0f) {
            // Get the measured onset time (in seconds) from Aubio
            double measured_time = aubio_onset_get_last_s(onset);
            // Smooth the measured time using the Kalman filter
            double smoothed_time = kalman_filter(measured_time);

            double predicted = smoothed_time;
            // If there is a previous event, predict the next onset time using AR(1)
            if (event_count > 0) {
                predicted = predict_next_onset(smoothed_time, events[event_count - 1].time);
            }

            // Lock mutex to safely update the events array
            pthread_mutex_lock(&event_mutex);
            if (event_count < MAX_EVENTS) {
                events[event_count].time = smoothed_time;
                events[event_count].pitch = current_pitch;
                events[event_count].predicted_next = predicted;
                event_count++;
            }
            pthread_mutex_unlock(&event_mutex);

            printf("Detected onset at %.3f sec, pitch: %.2f Hz, predicted next: %.3f sec\n",
                   smoothed_time, current_pitch, predicted);
        }
    }

    // Free allocated buffers and Aubio objects
    del_fvec(pitch_output);
    del_fvec(onset_output);
    del_fvec(in_buffer);
    del_aubio_pitch(pitch);
    del_aubio_onset(onset);
    del_aubio_source(source);
    aubio_cleanup();

    pthread_exit(NULL);
}

/*
 * Function to generate the results JSON string using cJSON.
 * The returned string is allocated on the heap; caller must free it with free().
 */
const char* GetResultsJson() {
    cJSON *root = cJSON_CreateObject();
    if (!root) return NULL;
    cJSON_AddStringToObject(root, "file", "guitar_track.wav");
    cJSON *onsets = cJSON_CreateArray();
    cJSON_AddItemToObject(root, "onsets", onsets);

    for (int i = 0; i < event_count; i++) {
        cJSON *item = cJSON_CreateObject();
        cJSON_AddNumberToObject(item, "time", events[i].time);
        cJSON_AddNumberToObject(item, "pitch", events[i].pitch);
        cJSON_AddNumberToObject(item, "predicted_next", events[i].predicted_next);
        cJSON_AddItemToArray(onsets, item);
    }

    char *json_str = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    return json_str;  // Caller must free this string using free()
}

/*
 * Function to free the JSON string returned by GetResultsJson().
 */
void FreeResultsJson(const char *json) {
    if (json) free((void*)json);
}

/*
 * Function to load and analyze a WAV file.
 * trackIndex is intended for multi-track files (e.g., guitar track index), but here we process mono.
 * Returns 0 on success.
 */
int LoadAndAnalyzeWav(const char *filepath, int trackIndex) {
    event_count = 0; // Reset global event count
    TrackParams params;
    params.filepath = filepath;
    params.track_index = trackIndex;
    pthread_t thread;
    int ret = pthread_create(&thread, NULL, process_track, &params);
    if (ret != 0) {
        fprintf(stderr, "Error: Cannot create analysis thread\n");
        return ret;
    }
    pthread_join(thread, NULL);
    return 0;
}

/*
 * Main function for testing the program standalone.
 * Usage: onset_analyzer <wavfile>
 */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <wavfile>\n", argv[0]);
        return 1;
    }
    if (LoadAndAnalyzeWav(argv[1], 0) != 0) {
        fprintf(stderr, "Error during analysis\n");
        return 1;
    }
    const char* json = GetResultsJson();
    if (json) {
        printf("Analysis Results:\n%s\n", json);
        FreeResultsJson(json);
    }
    return 0;
}
