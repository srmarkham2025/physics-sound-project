import os
import time
import matplotlib
matplotlib.use("Agg")
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import find_peaks
import math

specs = {}
audioCache = {}
audioSr = {}
instruments = {
    "piano": "sounds/piano.ogg",
    "violin": "sounds/violin.ogg",
    "flute": "sounds/flute.ogg",
    "clarinet": "sounds/clarinet.ogg"
}
ranges = {
    "piano": (27.5, 5000),
    "violin": (196, 5000),
    "flute": (261.63, 5000),
    "clarinet": (146.83, 5000)
}

def loadAudio(instrument_name, file_path):
    y, sr = librosa.load(file_path)
    audioCache[instrument_name] = y
    audioSr[instrument_name] = sr   

def generateSpec(instrument_name, file_path):
    save_path = f"static/{instrument_name.capitalize()}Spec.png"
    if os.path.exists(save_path):
        return f"/static/{instrument_name.capitalize()}Spec.png"
    y= audioCache[instrument_name]
    sr= audioSr[instrument_name]
    FT = librosa.stft(y, n_fft=2048)  # updated n_fft
    dB = librosa.amplitude_to_db(np.abs(FT), ref = np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(dB, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title=f"{instrument_name.capitalize()} Frequency-Time")
    plt.colorbar(img, ax = ax)
    save_path= f"static/{instrument_name.capitalize()}Spec.png"
    plt.savefig(save_path)
    plt.close()
    return f"{instrument_name.capitalize()}Spec.png"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.on_event("startup")
def startup_event():
    for name, path in instruments.items():
        loadAudio(name, path)
        spec_file = generateSpec(name, path)
        specs[name] = spec_file
        print(specs)

@app.get("/spectrogram")
def get_spectrograms():
    return specs

@app.get("/series")
def seriesCalculation(time: float):
    series = {}
    for name, path in instruments.items():
        if name != "piano":
            series[name] = generateSeries(name, path, time)
    return series

def generateSeries(instrument_name, file_path, time):
    y= audioCache[instrument_name]
    sr= audioSr[instrument_name]
    FT = librosa.stft(y, n_fft=2048)  # updated n_fft
    mag = np.abs(FT)   
    hop = 512
    frame = int(time * sr / hop)
    frame = min(frame, mag.shape[1] - 1)
    print("Frame:", frame)
    ampslice = mag[:, frame]
    
    peaks, _ = find_peaks(ampslice, height=np.max(ampslice)*0.075)
    
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)  # updated n_fft
    peakfreqs = frequencies[peaks]

    if len(peakfreqs) == 0:
        return []
    
    Amps = ampslice[peaks]
    fundamental = findFundamental(peakfreqs, Amps, instrument_name)
    overtones = [fundamental * (n+1) for n in range(len(peakfreqs))]
    equation = ""
    usedpeaks = []
    usedamps = []
    for i, overtone in enumerate(overtones):
        index = 0
        lowestSeparation = 99999
        for j, freq in enumerate(peakfreqs):
            if peakfreqs[j] not in usedpeaks:
                if np.abs(freq - overtone) < lowestSeparation:
                    lowestSeparation = np.abs(freq - overtone)
                    index = j
        if abs(peakfreqs[index] - overtone) < 50 :
            usedpeaks.append(peakfreqs[index])
            usedamps.append(Amps[index])

    for i, (A, f) in enumerate(zip(usedamps, usedpeaks)):
        if usedpeaks[i] < ranges[instrument_name][0] or usedpeaks[i] > ranges[instrument_name][1]:
            continue
        if equation != "":
            equation += " + "
        equation += f"{A:.3f} * sin(2*pi*{f:.3f}*t)"
    
    print("Frequencies:", peakfreqs)
    print("Amplitudes:", Amps)

    t = np.linspace(0, 0.01, 10000)  
    y=[0] * len(t)
    for j, tj in enumerate(t):
        for i, (amp, freq) in enumerate(zip(usedamps, usedpeaks)):
            y[j] += amp * math.sin(2*math.pi * tj* freq)
    plt.figure(figsize=(15,5))
    plt.plot(t, y)
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure (arbitrary units)")
    plt.title(f"{instrument_name.capitalize()} Pressure waveform")
    plt.grid(True)
    save_path= f"static/{instrument_name.capitalize()}Series.png"
    plt.savefig(save_path)
    plt.close()
    return {"equation": equation, "file": f"{instrument_name.capitalize()}Series.png"}

@app.get("/deviation")
def devCalculation(time: float):
    deviations = {}
    for name, path in instruments.items():
        if name != "piano":
            deviations[name] = generateDev(name, path, time)
    return deviations

def generateDev(instrument_name, file_path, time):
    y= audioCache[instrument_name]
    sr= audioSr[instrument_name]
    FT = librosa.stft(y, n_fft=2048)  # updated n_fft
    mag = np.abs(FT)   
    hop = 512
    frame = int(time * sr / hop)
    frame = min(frame, mag.shape[1] - 1)
    ampslice = mag[:, frame]
    
    peaks, _ = find_peaks(ampslice, height=np.max(ampslice)*0.075)
    
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)  # updated n_fft
    peakfreqs = frequencies[peaks]
    Amps = ampslice[peaks]
    overtones = []

    if len(peakfreqs) == 0:
        return []
    
    print(peakfreqs)

    fundamental = findFundamental(peakfreqs, Amps, instrument_name)
    overtones = [fundamental * (n+1) for n in range(len(peakfreqs))]

    print("Peak Frequencies:" )
    print(peakfreqs)
    print("Overtones:")
    print(overtones)

    deviation = []
    n = min(len(overtones), len(peakfreqs))
    missingOvertones=[]
    geometryRestrictedOvertones=[]
    usedpeaks = []
    for i, overtone in enumerate(overtones):
        index = 0
        lowestSeparation = 99999
        for j, freq in enumerate(peakfreqs):
            if peakfreqs[j] not in usedpeaks:
                if np.abs(freq - overtone) < lowestSeparation:
                    lowestSeparation = np.abs(freq - overtone)
                    index = j
            
        if abs(peakfreqs[index] - overtone) > 50 :
            deviation.append(-1)
            usedpeaks.append(-1)
            if abs((peakfreqs[index-1] - overtones[i-1])) < 50:
                geometryRestrictedOvertones.append((i+1, deviation[i]))
            else:
                missingOvertones.append((i+1, deviation[i]))
            
        else:
            deviation.append(peakfreqs[index] - overtone)
            usedpeaks.append(peakfreqs[index])

    x = np.linspace(1, n, n)

    print("Deviations:" )
    print(deviation)

    plt.scatter(x, deviation)
    print("Missing Overtones:")
    print(missingOvertones)
    print("Geometry Restricted Overtones:")
    print(geometryRestrictedOvertones)
    if missingOvertones:
        xMissing, yMissing = zip(*missingOvertones)
        plt.scatter(xMissing, yMissing, color='red', label='Missing Overtones', zorder=5)
    if geometryRestrictedOvertones:
        xGeo, yGeo = zip(*geometryRestrictedOvertones)
        plt.scatter(xGeo, yGeo, color='green', label='Geometry Restricted Overtones', zorder=6)

    plt.title(f"Overtone Deviation for {instrument_name.capitalize()}")
    plt.xlabel("Harmonic Number")
    plt.ylabel("Deviation (Hz)")
    plt.xlim(1, 10) 
    plt.ylim(-100, 100) 
    plt.legend()
    save_path= f"static/{instrument_name.capitalize()}Deviation.png"
    plt.savefig(save_path)
    plt.close()

    cellText = []
    for i in range(n):
        cellText.append([
            str(i+1), 
            f"{overtones[i]:.2f}", 
            f"{usedpeaks[i]:.2f}", 
            f"{deviation[i]:.2f}"
        ])

    colLabels = ["n", r"$f_{predicted} (Hz)$", r"$f_{measured}$ (Hz)", "Deviation (Hz)"]    
    fig, ax = plt.subplots(figsize=(6, max(2, n*0.4)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=cellText, colLabels=colLabels, loc='center', cellLoc='center')
    ax.set_title(f"{instrument_name.capitalize()} Overtone Deviation Table", fontweight="bold") 
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    table_path = f"static/{instrument_name}OvertoneTable.png"
    plt.savefig(table_path, bbox_inches='tight')
    plt.close()

    return {"file": f"{instrument_name.capitalize()}Deviation.png", "table": f"{instrument_name}OvertoneTable.png"}

def findFundamental(peakfreqs, amps, instrument):
    freqs = []
    freqamps = []
    minFreq= ranges[instrument][0]
    maxFreq= ranges[instrument][1]
    for i, freq in enumerate(peakfreqs):
        for j in range(12):
            if peakfreqs[i]/(j+1) < minFreq or peakfreqs[i]/(j+1) > maxFreq:
                continue
            freqs.append(peakfreqs[i]/(j+1))
            freqamps.append(amps[i])
    print(freqs)
    print(freqamps)
    freqs = np.array(freqs)
    freqamps = np.array(freqamps)
    tolerance = 10
    support = []
    for i, freq in enumerate(freqs):
        freqSupport = 0
        for f, amp in zip(peakfreqs, amps):
            n = round(f / freq)
            predicted = n * freq  
            if abs(f - predicted) <= tolerance:
                freqSupport += amp
        support.append(freqSupport)
    fundamental=freqs[np.argmax(support)]
    print("Fundamental Support:", support)
    print("Chosen Fundamental:", fundamental)
    return fundamental

