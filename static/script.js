document
  .getElementById('predictionForm')
  .addEventListener('submit', async function (event) {
    event.preventDefault();

    // Show loading state
    document.getElementById('hasilPrediksi').innerText = 'Memproses...';

    const data = {
      jumlah_penduduk: Number(document.getElementById('jumlah_penduduk').value),
      jumlah_kepadatan: Number(
        document.getElementById('jumlah_kepadatan').value
      ),
      pm_sepuluh: Number(document.getElementById('pm_sepuluh').value),
      pm_duakomalima: Number(document.getElementById('pm_duakomalima').value),
      sulfur_dioksida: Number(document.getElementById('sulfur_dioksida').value),
      karbon_monoksida: Number(
        document.getElementById('karbon_monoksida').value
      ),
      ozon: Number(document.getElementById('ozon').value),
      nitrogen_dioksida: Number(
        document.getElementById('nitrogen_dioksida').value
      ),
      wilayah: Number(document.getElementById('wilayah').value),
    };

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      // Create a more informative result display
      let resultHTML = `
        <div class="prediction-result">
          <h3>Hasil Analisis Kualitas Udara</h3>
          <p class="status ${result.health_status
            .toLowerCase()
            .replace(' ', '-')}">
            Status: <strong>${result.health_status}</strong>
          </p>
          <p class="description">${result.description}</p>
        </div>
      `;

      document.getElementById('hasilPrediksi').innerHTML = resultHTML;
    } catch (error) {
      document.getElementById('hasilPrediksi').innerText =
        'Terjadi kesalahan saat memproses permintaan: ' + error.message;
    }
  });
