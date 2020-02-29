const jpegThumbnail = dataUrl => new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        const maxDimension = 96; // 3x the maximum displayed size of 32px

        if (image.height > image.width) {
            canvas.height = maxDimension;
            canvas.width = (maxDimension / image.height) * image.width;
        } else {
            canvas.width = maxDimension;
            canvas.height = (maxDimension / image.width) * image.height;
        }

        ctx.fillStyle = 'white'; // Create white background, since jpeg doesn't have transparency
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL('image/jpeg', 0.92 /* quality */)); // Default quality is 0.92
    };
    image.onerror = err => {
        reject(err);
    };
    image.src = dataUrl;
});

export default jpegThumbnail;
