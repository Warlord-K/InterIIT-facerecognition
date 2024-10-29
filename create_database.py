import os
import cv2
from face_recognition_system import FaceRecognitionSystem
from pathlib import Path

class DatabaseBuilder:
    def __init__(self, base_folder, database_path="face_database.pkl"):
        """
        Initialize the database builder
        
        Args:
            base_folder (str): Path to the base folder containing institution folders
            database_path (str): Path where the face database will be saved
        """
        self.base_folder = Path(base_folder)
        self.face_system = FaceRecognitionSystem(database_path=database_path)
        self.supported_extensions = {'.jpg', '.jpeg', '.png'}
        
    def process_name(self, filename, institution):
        """
        Process the filename to create a standardized name format
        
        Args:
            filename (str): Original filename (e.g., 'Abhinav_Kumar.jpg')
            institution (str): Institution name (e.g., 'IITI')
            
        Returns:
            str: Formatted name (e.g., 'AbhinavK_IITI')
        """
        # Remove file extension
        name = Path(filename).stem
        
        # Split name parts
        parts = name.replace('-', '_').split('_')
        
        # Take first letter of last name and full first name
        if len(parts) >= 2:
            formatted_name = f"{parts[0]}{parts[1][0]}"
        else:
            formatted_name = parts[0]
            
        # Combine with institution
        return f"{formatted_name}_{institution}"
    
    def build_database(self):
        """Build the face database from the folder structure"""
        if not self.base_folder.exists():
            raise FileNotFoundError(f"Base folder {self.base_folder} not found")
            
        # Track statistics
        processed = 0
        failed = 0
        
        # Process each institution folder
        for institution_folder in self.base_folder.iterdir():
            if not institution_folder.is_dir():
                continue
                
            institution = institution_folder.name
            print(f"\nProcessing institution: {institution}")
            
            # Process each image in the institution folder
            for image_path in institution_folder.iterdir():
                if image_path.suffix.lower() not in self.supported_extensions:
                    continue
                    
                try:
                    # Read image
                    frame = cv2.imread(str(image_path))
                    if frame is None:
                        print(f"Failed to read image: {image_path}")
                        failed += 1
                        continue
                        
                    # Process name
                    formatted_name = self.process_name(image_path.name, institution)
                    
                    # Add to database
                    success = self.face_system.add_face_to_database(formatted_name, frame)
                    
                    if success:
                        processed += 1
                        print(f"Successfully processed: {formatted_name}")
                    else:
                        failed += 1
                        print(f"Failed to process: {image_path}")
                        
                except Exception as e:
                    failed += 1
                    print(f"Error processing {image_path}: {e}")
                    
        # Print summary
        print(f"\nDatabase building complete!")
        print(f"Successfully processed: {processed} images")
        print(f"Failed to process: {failed} images")
        print(f"Total images attempted: {processed + failed}")

def main():
    # You can modify these paths as needed
    base_folder = "./"
    database_path = "face_database.pkl"
    
    try:
        builder = DatabaseBuilder(base_folder, database_path)
        builder.build_database()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()