import model_finetuned
import model_random_weights
import predict_test_data

def main():
    print("\n================================================")
    print("Koulutetaan kaikki raportoidut mallit alusta...")
    print("==================================================")
    
    print("Ajetaan Resnet18 mallin finetuunaus...")
    model_finetuned.main()
    print("================================================\n")
    
    print("\n================================================")
    print("Ajetaan random weight mallin koulutus (lr=0.001, threshold=0.7)...")
    model_random_weights.main(lr=0.001, threshold=0.7)
    print("================================================\n")

    print("\n================================================")
    print("Ajetaan random weight mallin koulutus (lr=0.005, threshold=0.75)...")
    model_random_weights.main(lr=0.005, threshold=0.75)
    print("================================================\n")

    print("\n================================================")
    print("Ennustetaan treenidatasta samoin kuin tulevasta testidatasta...")
    predict_test_data.main()
    print("================================================\n")
    
if __name__ == "__main__":
    main()