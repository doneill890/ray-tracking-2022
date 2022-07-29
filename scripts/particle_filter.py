import numpy as np
import matplotlib.pyplot as plt
norm = np.linalg.norm

class ParticleFilter():

    def __init__(self, start):
        ''' initializes a new particle filter with the set of particles around the start point '''
        self._num_particles = 1000
        self._size = 3
        x, y = start
        box = 20
        particles = np.zeros((self._num_particles, self._size))
        particles[:, 0] = np.random.uniform(x-box, x+box, self._num_particles)
        particles[:, 1] = np.random.uniform(y-box, y+box, self._num_particles)
        self._particles = particles
        self._meas_var = 15
        self._last_est = start
    
    def _gaussian(self, x):
        ''' helper function, just defines a gaussian districution '''
        stdev = self._meas_var
        coeff = 1/(stdev*np.sqrt(2*np.pi))
        exponent = -.5*(x/stdev)**2
        return coeff*np.exp(exponent)
    
    def _get_mean(self):
        ''' returns the mean of the current particle set '''
        return np.mean(self._particles, axis=0)[0:-1]
    
    def _predict(self):
        ''' prediction step, updates particles assuming constant velocity with some noise '''
        d_pos = self._get_mean() - self._last_est
        predicted_particles = np.zeros((self._num_particles, self._size))
        for i in range(0, self._num_particles):
            particle_old = self._particles[i][0:self._size-1] # take off the weight
            particle_noise = np.array([np.random.normal(0, 2), np.random.normal(0, 2)])
            particle_pred = particle_old + particle_noise
            predicted_particles[i] = np.concatenate((particle_pred, np.array([0])))
        self._particles = predicted_particles
    
    def _correct(self, measurement):
        ''' correction step once a measurement comes in '''
        self._last_est = self._get_mean()                                           # update previous estimate

        for i in range(0, self._num_particles):                                     # assign weights
            particle_pred = self._particles[i][0:self._size-1]
            weight = np.array([self._gaussian(norm(particle_pred - measurement))])
            self._particles[i][-1] =  weight

        corrected_particles = np.zeros((self._num_particles, self._size))           # resample according to weights
        min_weight = 1e-60
        self._particles[:, -1] = np.maximum(self._particles[:, -1], min_weight)
        w_tot = np.sum(self._particles[:, -1])
        probs = self._particles[:, -1] = self._particles[:, -1] / w_tot
        indices = np.random.choice(len(self._particles), len(self._particles), p=probs)
        corrected_particles = self._particles[indices]
        self._particles = corrected_particles                                       # update particles

    def _plot(self):
        ''' plots all the particles '''
        plt.scatter(self._particles[:,0], self._particles[:,1])
        plt.xlim((-150, 150))
        plt.ylim((-150, 150))
        plt.pause(0.1)
        plt.clf()

if __name__ == '__main__':
    ''' simple particle filter test run '''
    pf = ParticleFilter(np.array([0, 0]))
    meas = np.array([0, 0])
    plt.ion()
    for i in range(0, 20):
        pf._predict(meas)
        pf._correct()
        pf._plot()
