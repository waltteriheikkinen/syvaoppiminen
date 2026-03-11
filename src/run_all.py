import model
import predict_with_model

def main():
    print("Aloitetaan mallin koulutus...")
    model.main()
    
    print("\nAloitetaan ennustus koulutetulla mallilla...")
    predict_with_model.main()
    

if __name__ == "__main__":
    main()