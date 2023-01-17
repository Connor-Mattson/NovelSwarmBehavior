import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if __name__ == "__main__":

    a_x = range(1, 50)
    a_y = [0.674597652063, 0.47994815497446913, 0.3587211031527827, 0.37505628028855315, 0.4441278672367998, 0.39136982942282367, 0.33930266384282765, 0.32310382690982453, 0.292688118459703, 0.27636341962987737, 0.2926413020542609, 0.28612740819456356, 0.3019182320796824, 0.3366658119024394, 0.3136027979873262, 0.2966352551971198, 0.24729064822674843, 0.24871836679364748, 0.22395041406157928, 0.21595073802292014, 0.22676402528228434, 0.22291881193255988, 0.20937319727560336, 0.22229061351182328, 0.20125891887960853, 0.22404431777973277, 0.2114004391472472, 0.20622674126886756, 0.2033473505529856, 0.18454270623277239, 0.22308763244712584, 0.21563170423206113, 0.21778874827188754, 0.20077347267711793, 0.19667778562447413, 0.19471457445818732, 0.18850319757156117, 0.2028551180575, 0.2007092487210296, 0.1942269032253961, 0.19336891910646284, 0.17795744881220055, 0.1695250312047799, 0.16362209518223103, 0.1654865120421185, 0.1749876323866703, 0.16324539527345017, 0.1666727685731074, 0.16983527676571963]

    b_x = range(1, 50)
    b_y = [1 for i in range(1,50)]

    plt.plot(a_x, a_y, label="Heuristic Filtering")
    plt.plot(b_x, b_y, label="Original")

    plt.title("Novelty Score Over Time")
    plt.xlabel("Generations of Novelty Search")
    plt.ylabel("Novelty Score")
    plt.legend()
    plt.show()