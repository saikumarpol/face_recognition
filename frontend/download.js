const fs = require('fs');
const https = require('https');
const path = require('path');

const downloadFile = (url, destination) => {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(destination);
    https.get(url, response => {
      response.pipe(file);
      file.on('finish', () => {
        file.close();
        resolve();
      });
    }).on('error', err => {
      fs.unlink(destination, () => {
        reject(err);
      });
    });
  });
};

async function downloadModels() {
  const modelsDir = path.join(process.cwd(), 'public', 'models');
  
  // Create models directory if it doesn't exist
  if (!fs.existsSync(modelsDir)) {
    fs.mkdirSync(modelsDir, { recursive: true });
  }

  const files = [
    {
      url: 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_expression_model-weights_manifest.json',
      path: path.join(modelsDir, 'face_expression_model-weights_manifest.json')
    },
    {
      url: 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_expression_model-shard1',
      path: path.join(modelsDir, 'face_expression_model-shard1')
    }
  ];

  console.log('Downloading face expression model files...');
  
  for (const file of files) {
    try {
      console.log(`Downloading ${path.basename(file.path)}...`);
      await downloadFile(file.url, file.path);
      console.log(`Successfully downloaded ${path.basename(file.path)}`);
    } catch (error) {
      console.error(`Error downloading ${path.basename(file.path)}:`, error);
    }
  }
}

downloadModels();