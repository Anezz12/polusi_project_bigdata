
document.getElementById('predictionForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const data = {
        jumlah_penduduk: Number(document.getElementById('jumlah_penduduk').value),
        jumlah_kepadatan: Number(document.getElementById('jumlah_kepadatan').value),
        sulfur_dioksida: Number(document.getElementById('sulfur_dioksida').value),
        karbon_monoksida: Number(document.getElementById('karbon_monoksida').value),
        ozon: Number(document.getElementById('ozon').value),
        nitrogen_dioksida: Number(document.getElementById('nitrogen_dioksida').value),
        wilayah: Number(document.getElementById('wilayah').value)
    };

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    });

    const result = await response.json();
    document.getElementById('hasilPrediksi').innerText = "Prediksi Parameter Pencemar Kritis: " + result.prediction;
});
